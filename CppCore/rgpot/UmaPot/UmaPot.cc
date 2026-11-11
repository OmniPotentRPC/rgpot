// MIT License — UmaPot: vesin neighbor list + AOTInductor .pt2

#include "rgpot/UmaPot/UmaPot.hpp"
#include "rgpot/UmaPot/MolecularFrame.hpp"

#include "rgpot/MetatomicPot/vesin_compat.hpp"
#include "rgpot/UmaPot/UmaContract.hpp"
#include "rgpot/UmaPot/aoti_execstack.hpp"
#include "vesin.h"

#include <array>
#include <vector>

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rgpot {
namespace {

bool ends_with(const std::string &s, const std::string &suf) {
  return s.size() >= suf.size() &&
         s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

torch::Device parse_device(const std::string &name) {
  if (name.empty() || name == "cpu")
    return torch::kCPU;
  if (name.rfind("cuda", 0) == 0)
    return torch::Device(name);
  return torch::kCPU;
}

struct EdgeList {
  std::vector<int64_t> neighbor;
  std::vector<int64_t> center;
  std::vector<double> offsets; // nedges * 3, cell shifts
};

EdgeList vesin_fairchem_edges(const ForceInput &in, double cutoff,
                              int max_neighbors) {
  VesinOptions options{};
  vesin_compat::fill_neighbor_options(options, cutoff, /*full_list=*/true);
  // One neighbor-list thread: the pair order then depends on the input alone.
  vesin_compat::set_n_threads(options, 1);
  options.return_distances = true;

  VesinNeighborList nl{};
  VesinDevice cpu = vesin_compat::make_cpu_device();
  const char *err = nullptr;
  bool pbc[3] = {true, true, true};
  const int status =
      vesin_neighbors(reinterpret_cast<const double(*)[3]>(in.pos), in.nAtoms,
                      reinterpret_cast<const double(*)[3]>(in.box), pbc, cpu,
                      options, &nl, &err);
  if (status != EXIT_SUCCESS) {
    std::string msg = "UmaPot: vesin_neighbors failed";
    if (err)
      msg += ": " + std::string(err);
    vesin_free(&nl);
    throw std::runtime_error(msg);
  }

  struct Cand {
    int64_t n{0};
    int64_t c{0};
    double s0{0};
    double s1{0};
    double s2{0};
    double dist{0};
  };
  std::vector<Cand> keep;
  keep.reserve(nl.length);
  const double *box = in.box;
  for (size_t k = 0; k < nl.length; ++k) {
    const int64_t i = static_cast<int64_t>(nl.pairs[k][0]);
    const int64_t j = static_cast<int64_t>(nl.pairs[k][1]);
    double dist = 0.0;
    if (nl.distances != nullptr) {
      dist = nl.distances[k];
    } else {
      const double sx = static_cast<double>(nl.shifts[k][0]);
      const double sy = static_cast<double>(nl.shifts[k][1]);
      const double sz = static_cast<double>(nl.shifts[k][2]);
      const double dx = in.pos[3 * j] - in.pos[3 * i] + sx * box[0] +
                        sy * box[3] + sz * box[6];
      const double dy = in.pos[3 * j + 1] - in.pos[3 * i + 1] + sx * box[1] +
                        sy * box[4] + sz * box[7];
      const double dz = in.pos[3 * j + 2] - in.pos[3 * i + 2] + sx * box[2] +
                        sy * box[5] + sz * box[8];
      dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    if (dist < 1e-8)
      continue;
    Cand c;
    c.n = j;
    c.c = i;
    c.s0 = static_cast<double>(nl.shifts[k][0]);
    c.s1 = static_cast<double>(nl.shifts[k][1]);
    c.s2 = static_cast<double>(nl.shifts[k][2]);
    c.dist = dist;
    keep.push_back(c);
  }
  vesin_free(&nl);

  // Canonical edge order: the neighbor list may be assembled by several
  // threads, so its pair order is not a function of the input. Message
  // passing sums follow the edge order, so a run is reproducible only if
  // this order is. Total order on (center, distance, neighbor, shift).
  std::sort(keep.begin(), keep.end(), [](const Cand &a, const Cand &b) {
    if (a.c != b.c)
      return a.c < b.c;
    if (a.dist != b.dist)
      return a.dist < b.dist;
    if (a.n != b.n)
      return a.n < b.n;
    if (a.s0 != b.s0)
      return a.s0 < b.s0;
    if (a.s1 != b.s1)
      return a.s1 < b.s1;
    return a.s2 < b.s2;
  });
  if (max_neighbors > 0) {
    std::vector<std::vector<size_t>> by_center(in.nAtoms);
    for (size_t k = 0; k < keep.size(); ++k)
      by_center[static_cast<size_t>(keep[k].c)].push_back(k);
    std::vector<Cand> trimmed;
    trimmed.reserve(keep.size());
    for (auto &idxs : by_center) {
      std::sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b) {
        return keep[a].dist < keep[b].dist;
      });
      if (static_cast<int>(idxs.size()) > max_neighbors)
        idxs.resize(static_cast<size_t>(max_neighbors));
      for (size_t id : idxs)
        trimmed.push_back(keep[id]);
    }
    keep.swap(trimmed);
  }

  EdgeList out;
  out.neighbor.reserve(keep.size());
  out.center.reserve(keep.size());
  out.offsets.reserve(keep.size() * 3);
  for (const auto &c : keep) {
    out.neighbor.push_back(c.n);
    out.center.push_back(c.c);
    out.offsets.push_back(c.s0);
    out.offsets.push_back(c.s1);
    out.offsets.push_back(c.s2);
  }
  return out;
}

const char *dtype_token(torch::ScalarType dt) {
  if (dt == torch::kFloat64)
    return "float64";
  if (dt == torch::kFloat32)
    return "float32";
  return "unknown";
}

std::optional<torch::ScalarType> dtype_from_name(const std::string &raw) {
  if (raw == "float64" || raw == "double" || raw == "torch.float64")
    return torch::kFloat64;
  if (raw == "float32" || raw == "float" || raw == "torch.float32")
    return torch::kFloat32;
  return std::nullopt;
}

// Every edge index must land inside the position buffer. The compiled
// graph gathers by this index, so a value past the last atom would
// read off the end of that buffer.
void reject_periodic_images(const std::vector<double> &offsets) {
  for (double shift : offsets) {
    if (shift == 0.0)
      continue;
    throw UmaContractError(
        "cell_offsets", "a molecular box with no periodic image", "nonzero",
        "input");
  }
}

// One edge-count contract for a single system and for every system in
// a band. The compiled graph is not called when this throws.
void apply_edge_contract(const UmaContract &contract, bool molecular,
                         int64_t natoms, int64_t systems, const int64_t *counts,
                         int64_t nedges_total) {
  contract.checkEachSystem(systems, counts, natoms, molecular);
  if (contract.pos_decl.known || contract.edge_decl.known)
    contract.checkCallShapes(natoms, nedges_total);
  else
    contract.checkEdges(nedges_total);
}

void check_edge_indices(const torch::Tensor &edge_index, int64_t npos) {
  const auto ei = edge_index.accessor<int64_t, 2>();
  const int64_t nedges = edge_index.size(1);
  for (int64_t e = 0; e < nedges; ++e) {
    for (int row = 0; row < 2; ++row) {
      const int64_t v = ei[row][e];
      if (v < 0 || v >= npos) {
        throw UmaShapeError(
            "edge_index", "[2, " + std::to_string(nedges) + "]",
            "pos [" + std::to_string(npos) + ", 3] (index " +
                std::to_string(v) + " is outside it)");
      }
    }
  }
}

// Write host doubles into a tensor that already has the package dtype.
void store_rows(torch::Tensor dst, const double *src, int64_t rows,
                int64_t cols) {
  if (dst.scalar_type() == torch::kFloat64) {
    auto a = dst.accessor<double, 2>();
    for (int64_t i = 0; i < rows; ++i)
      for (int64_t c = 0; c < cols; ++c)
        a[i][c] = src[static_cast<size_t>(i * cols + c)];
    return;
  }
  if (dst.scalar_type() == torch::kFloat32) {
    auto a = dst.accessor<float, 2>();
    for (int64_t i = 0; i < rows; ++i)
      for (int64_t c = 0; c < cols; ++c)
        a[i][c] =
            static_cast<float>(src[static_cast<size_t>(i * cols + c)]);
    return;
  }
  throw std::runtime_error(
      "UmaPot: floating input dtype is not float32 or float64");
}

double edge_span(const ForceInput &in, int64_t neighbor, int64_t center,
                 double s0, double s1, double s2) {
  const double *box = in.box;
  const double dx = in.pos[3 * neighbor] - in.pos[3 * center] + s0 * box[0] +
                    s1 * box[3] + s2 * box[6];
  const double dy = in.pos[3 * neighbor + 1] - in.pos[3 * center + 1] +
                    s0 * box[1] + s1 * box[4] + s2 * box[7];
  const double dz = in.pos[3 * neighbor + 2] - in.pos[3 * center + 2] +
                    s0 * box[2] + s1 * box[5] + s2 * box[8];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Every ordered pair, shift 0, ordered by (center, distance, neighbor).
// A molecular package is traced at this count. The metadata cutoff is the
// device that was supposed to produce it (0.4 of the box, 10 angstrom when
// the box is 25) and drops a longer pair.
EdgeList complete_ordered_edges(const double *pos, size_t n) {
  struct Cand {
    int64_t neighbor;
    int64_t center;
    double dist;
  };
  std::vector<Cand> keep;
  if (n > 1)
    keep.reserve(n * (n - 1));
  for (size_t c = 0; c < n; ++c) {
    for (size_t j = 0; j < n; ++j) {
      if (c == j)
        continue;
      const double dx = pos[3 * j] - pos[3 * c];
      const double dy = pos[3 * j + 1] - pos[3 * c + 1];
      const double dz = pos[3 * j + 2] - pos[3 * c + 2];
      Cand row;
      row.neighbor = static_cast<int64_t>(j);
      row.center = static_cast<int64_t>(c);
      row.dist = std::sqrt(dx * dx + dy * dy + dz * dz);
      keep.push_back(row);
    }
  }
  std::sort(keep.begin(), keep.end(), [](const Cand &a, const Cand &b) {
    if (a.center != b.center)
      return a.center < b.center;
    if (a.dist != b.dist)
      return a.dist < b.dist;
    return a.neighbor < b.neighbor;
  });
  EdgeList out;
  out.neighbor.reserve(keep.size());
  out.center.reserve(keep.size());
  out.offsets.reserve(keep.size() * 3);
  for (const auto &row : keep) {
    out.neighbor.push_back(row.neighbor);
    out.center.push_back(row.center);
    out.offsets.push_back(0.f);
    out.offsets.push_back(0.f);
    out.offsets.push_back(0.f);
  }
  return out;
}

// Static packages record one edge count. A short list is filled with an
// image of atoms 0 and 1 whose distance sits past `range`, so the model
// envelope contributes nothing. Edges already past `range` are dropped
// when the list is long; an in-range overflow is left for checkEdges.
EdgeList fit_edge_count(const ForceInput &in, EdgeList edges, int64_t want,
                        double range) {
  const int64_t have = static_cast<int64_t>(edges.neighbor.size());
  if (want <= 0 || have == want)
    return edges;
  if (have > want) {
    std::vector<double> dist(static_cast<size_t>(have));
    std::vector<size_t> order(static_cast<size_t>(have));
    for (int64_t e = 0; e < have; ++e) {
      dist[static_cast<size_t>(e)] = edge_span(
          in, edges.neighbor[static_cast<size_t>(e)],
          edges.center[static_cast<size_t>(e)],
          edges.offsets[static_cast<size_t>(3 * e)],
          edges.offsets[static_cast<size_t>(3 * e + 1)],
          edges.offsets[static_cast<size_t>(3 * e + 2)]);
      order[static_cast<size_t>(e)] = static_cast<size_t>(e);
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return dist[a] > dist[b];
    });
    std::vector<char> drop(static_cast<size_t>(have), 0);
    int64_t kept = have;
    for (size_t idx : order) {
      if (kept <= want)
        break;
      if (dist[idx] >= range) {
        drop[idx] = 1;
        --kept;
      }
    }
    EdgeList trimmed;
    trimmed.neighbor.reserve(static_cast<size_t>(kept));
    trimmed.center.reserve(static_cast<size_t>(kept));
    trimmed.offsets.reserve(static_cast<size_t>(kept) * 3);
    for (int64_t e = 0; e < have; ++e) {
      if (drop[static_cast<size_t>(e)])
        continue;
      trimmed.neighbor.push_back(edges.neighbor[static_cast<size_t>(e)]);
      trimmed.center.push_back(edges.center[static_cast<size_t>(e)]);
      trimmed.offsets.push_back(edges.offsets[static_cast<size_t>(3 * e)]);
      trimmed.offsets.push_back(edges.offsets[static_cast<size_t>(3 * e + 1)]);
      trimmed.offsets.push_back(edges.offsets[static_cast<size_t>(3 * e + 2)]);
    }
    return trimmed;
  }
  if (in.nAtoms < 2)
    throw std::runtime_error(
        "UmaPot: static edge count needs at least two atoms to pad");
  // One periodic image of atoms 0 and 1. Along a cell edge of length L,
  // |s*a + d| >= s*L - |d|, so the image lies past `range` once
  // s > (range + |d|) / L. The longest edge keeps s small. ceil can
  // land on equality, and one further step clears that boundary.
  const double pair = edge_span(in, 1, 0, 0.0, 0.0, 0.0);
  const int column[3] = {0, 3, 6};
  int axis = -1;
  double longest = 0.0;
  for (int a = 0; a < 3; ++a) {
    const double x = in.box[column[a]];
    const double y = in.box[column[a] + 1];
    const double z = in.box[column[a] + 2];
    const double len = std::sqrt(x * x + y * y + z * z);
    if (len > longest) {
      longest = len;
      axis = a;
    }
  }
  if (!(longest > 0.0))
    throw std::runtime_error("UmaPot: cannot pad an edge in a zero cell");
  int shift = static_cast<int>(std::ceil((range + pair) / longest));
  if (shift < 1)
    shift = 1;
  auto span_at = [&](int s) {
    const double v = static_cast<double>(s);
    if (axis == 0)
      return edge_span(in, 1, 0, v, 0.0, 0.0);
    if (axis == 1)
      return edge_span(in, 1, 0, 0.0, v, 0.0);
    return edge_span(in, 1, 0, 0.0, 0.0, v);
  };
  if (!(span_at(shift) > range))
    ++shift;
  if (!(span_at(shift) > range))
    throw std::runtime_error(
        "UmaPot: cannot pad an edge past the model cutoff");
  // cell_offsets are float32. Integers through 2^24 stay exact.
  constexpr int kMaxExactFloatShift = 1 << 24;
  if (shift > kMaxExactFloatShift)
    throw std::runtime_error(
        "UmaPot: padded image shift is not an exact float32 integer");
  const float s0 = axis == 0 ? static_cast<float>(shift) : 0.f;
  const float s1 = axis == 1 ? static_cast<float>(shift) : 0.f;
  const float s2 = axis == 2 ? static_cast<float>(shift) : 0.f;
  edges.neighbor.reserve(static_cast<size_t>(want));
  edges.center.reserve(static_cast<size_t>(want));
  edges.offsets.reserve(static_cast<size_t>(want) * 3);
  while (static_cast<int64_t>(edges.neighbor.size()) < want) {
    edges.neighbor.push_back(1);
    edges.center.push_back(0);
    edges.offsets.push_back(s0);
    edges.offsets.push_back(s1);
    edges.offsets.push_back(s2);
  }
  return edges;
}

// Equal nedges_min and nedges_max are one compiled count. A band package
// records the total, so the per-system count is that total divided by the
// system count. A range, or a missing key, returns 0.
int64_t static_edges_per_system(const UmaContract &contract,
                                int64_t n_systems) {
  if (!contract.nedges_min || !contract.nedges_max ||
      *contract.nedges_min != *contract.nedges_max || n_systems < 1)
    return 0;
  const int64_t total = *contract.nedges_max;
  if (n_systems == 1)
    return total;
  if (total % n_systems != 0)
    return total;
  return total / n_systems;
}

EdgeList build_system_edges(const ForceInput &local, double cutoff,
                            double model_cutoff, double molecular_box,
                            int max_neighbors, const UmaContract &contract,
                            int64_t n_systems) {
  const int64_t n = static_cast<int64_t>(local.nAtoms);
  const int64_t complete = n > 1 ? n * (n - 1) : 0;
  const int64_t want = static_edges_per_system(contract, n_systems);
  const bool molecular = molecular_box > 0.0;
  // The compiled shape is every ordered pair: either the recorded count
  // says so, or a molecular package left the count unrecorded (the trace
  // contract is n(n-1)). The package cutoff is not applied. Pairs past
  // the model's own range stay in the list and the envelope zeroes them.
  if (complete > 0 && (want == complete || (molecular && want == 0)))
    return complete_ordered_edges(local.pos, local.nAtoms);

  const double range = (model_cutoff > 0.0 && std::isfinite(model_cutoff))
                           ? model_cutoff
                           : cutoff;
  EdgeList edges = vesin_fairchem_edges(local, range, max_neighbors);
  if (want > 0 && static_cast<int64_t>(edges.neighbor.size()) != want)
    edges = fit_edge_count(local, std::move(edges), want, range);
  return edges;
}

} // namespace

struct UmaPot::Impl {
  torch::Device device{torch::kCPU};
  // Input dtype of pos, cell and cell_offsets. float32 until metadata
  // says otherwise, including packages that record neither key.
  torch::ScalarType dtype{torch::kFloat32};
  std::string pos_dtype_name{"float32"};
  std::string model_dtype_name{"float32"};
  double cutoff{6.0};
  // Backbone interaction range from metadata `model_cutoff`. 0 when the
  // package does not record it; `cutoff` is the fallback.
  double model_cutoff{0.0};
  double molecular_box{0.0};
  int max_neighbors{300};
  int64_t batch_max{0};
  UmaContract contract;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;
  mutable std::mutex mutex;
};

void UmaPot::recomputeParamsKey() {
  Fnv1a fp;
  fp.u64(kKernelVersion);
  fp.str(m_config.model_path);
  fp.str(m_config.task_name);
  fp.str(m_config.device);
  fp.i64(m_config.charge);
  fp.i64(m_config.spin);
  m_paramsKey = fp.h;
}

void UmaPot::ensureLoaded() const {
  if (m_impl->loader)
    return;
  c10::InferenceMode guard;
  const std::string load_path =
      aoti_execstack::prepare_pt2_for_load(m_config.model_path);
  m_impl->loader =
      std::make_unique<torch::inductor::AOTIModelPackageLoader>(load_path);
  // The package's embedded metadata is the runtime contract: one
  // producer (scripts/export_uma_aoti.py), the .pt2 self-describing.
  const auto meta = m_impl->loader->get_metadata();
  auto num = [&](const char *key, double fallback) {
    const auto it = meta.find(key);
    if (it == meta.end() || it->second.empty())
      return fallback;
    try {
      return std::stod(it->second);
    } catch (...) {
      return fallback;
    }
  };
  m_impl->cutoff = num("cutoff", m_impl->cutoff);
  m_impl->model_cutoff = num("model_cutoff", 0.0);
  if (!std::isfinite(m_impl->model_cutoff) || !(m_impl->model_cutoff > 0.0))
    m_impl->model_cutoff = 0.0;
  m_impl->molecular_box = num("molecular_box", m_impl->molecular_box);
  m_impl->max_neighbors = static_cast<int>(
      num("max_neighbors", static_cast<double>(m_impl->max_neighbors)));
  m_impl->batch_max = static_cast<int64_t>(num("batch_max", 0.0));
  auto named = [&](const char *key) -> std::optional<torch::ScalarType> {
    const auto it = meta.find(key);
    if (it == meta.end() || it->second.empty())
      return std::nullopt;
    const auto parsed = dtype_from_name(it->second);
    if (!parsed)
      throw std::runtime_error(std::string("UmaPot: package metadata ") +
                               key + " is not float32 or float64: " +
                               it->second);
    return parsed;
  };
  const auto pos = named("pos_dtype");
  const auto model = named("model_dtype");
  // Log only keys the package recorded. "unset" is not a guessed dtype.
  m_impl->pos_dtype_name = pos ? dtype_token(*pos) : "unset";
  m_impl->model_dtype_name = model ? dtype_token(*model) : "unset";
  if (pos && model && *pos != *model) {
    throw std::runtime_error(
        std::string("UmaPot: pos_dtype is ") + m_impl->pos_dtype_name +
        " and model_dtype is " + m_impl->model_dtype_name +
        "; refusing to feed one dtype into a graph recorded as the other");
  }
  if (!pos && !model) {
    m_impl->dtype = torch::kFloat32;
    std::fprintf(stderr,
                 "UmaPot: package metadata has no pos_dtype or model_dtype; "
                 "falling back to float32\n");
  } else {
    m_impl->dtype = pos ? *pos : *model;
  }
  std::fprintf(stderr, "UmaPot: pos_dtype=%s model_dtype=%s\n",
               m_impl->pos_dtype_name.c_str(),
               m_impl->model_dtype_name.c_str());
  m_impl->contract = UmaContract::fromMetadata(meta);
}

void UmaPot::checkContract(const ForceInput &in) const {
  m_impl->contract.checkConfig(m_config);
  m_impl->contract.checkSystem(in.nAtoms, in.atmnrs);
}

UmaPot::UmaPot(const UmaConfig &config)
    : Potential(PotType::Uma), m_config(config) {
  if (m_config.model_path.empty()) {
    throw std::runtime_error(
        "UmaPot: model_path is required (AOTI package .pt2)");
  }
  if (!ends_with(m_config.model_path, ".pt2")) {
    throw std::runtime_error(
        "UmaPot: model_path must be an AOTI .pt2 from "
        "scripts/export_uma_aoti.py (not a metatomic/torchscript .pt)");
  }
  recomputeParamsKey();
  m_impl = std::make_unique<Impl>();
  m_impl->device = parse_device(m_config.device);
  m_impl->max_neighbors = m_config.max_neighbors;
  m_impl->cutoff = m_config.cutoff;
  if (!(m_impl->cutoff > 0.0) || !std::isfinite(m_impl->cutoff))
    m_impl->cutoff = 6.0;
}

UmaPot::~UmaPot() = default;

void UmaPot::setChargeSpin(int charge, int spin) {
  m_config.charge = charge;
  m_config.spin = spin;
  recomputeParamsKey();
}

void UmaPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (in.nAtoms == 0)
    throw std::runtime_error("UmaPot: nAtoms == 0");
  {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    ensureLoaded();
  }
  checkContract(in);
  if (m_impl->batch_max > 1) {
    // A band package carries only the batched (B >= 2) graph; a lone
    // system rides the batch path, which pads it to two.
    ForceBatch one{.nSystems = 1, .in = &in, .out = out};
    forceBatchImpl(one);
    return;
  }

  // Molecular inputs use the sidecar's cube and an origin-centred frame.
  // A common translation preserves all relative vectors and cell offsets;
  // avoiding a large coordinate offset retains precision in float32 models.
  const bool molecular = m_impl->molecular_box > 0.0;
  std::vector<double> mol_pos;
  std::array<double, 9> mol_box{};
  if (molecular) {
    const double L = m_impl->molecular_box;
    mol_pos.assign(in.pos, in.pos + 3 * in.nAtoms);
    mol_box = uma::molecularFrame(mol_pos, L);
  }
  const ForceInput local{in.nAtoms, molecular ? mol_pos.data() : in.pos,
                         in.atmnrs, molecular ? mol_box.data() : in.box};

  const EdgeList edges = build_system_edges(
      local, m_impl->cutoff, m_impl->model_cutoff, m_impl->molecular_box,
      m_impl->max_neighbors, m_impl->contract, /*n_systems=*/1);
  const auto n = static_cast<int64_t>(local.nAtoms);
  const auto nedges = static_cast<int64_t>(edges.neighbor.size());
  if (nedges == 0)
    throw std::runtime_error("UmaPot: vesin produced no edges");
  if (molecular)
    reject_periodic_images(edges.offsets);
  // The same check the band path uses. A molecular count other than
  // n*(n-1), or a count outside the traced range, is named here.
  const int64_t one_count = nedges;
  apply_edge_contract(m_impl->contract, molecular, n, 1, &one_count, nedges);

  auto opts_f = torch::TensorOptions().dtype(m_impl->dtype).device(torch::kCPU);
  auto opts_l = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);

  auto pos = torch::empty({n, 3}, opts_f);
  store_rows(pos, local.pos, n, 3);

  auto atomic_numbers = torch::empty({n}, opts_l);
  auto z_a = atomic_numbers.accessor<int64_t, 1>();
  for (int64_t i = 0; i < n; ++i)
    z_a[i] = local.atmnrs[i];

  auto cell = torch::empty({1, 3, 3}, opts_f);
  store_rows(cell.view({1, 9}), local.box, 1, 9);
  cell = cell.view({1, 3, 3});

  auto pbc = torch::tensor({{true, true, true}},
                           torch::TensorOptions().dtype(torch::kBool));
  auto edge_index = torch::empty({2, nedges}, opts_l);
  auto ei = edge_index.accessor<int64_t, 2>();
  for (int64_t e = 0; e < nedges; ++e) {
    ei[0][e] = edges.neighbor[static_cast<size_t>(e)];
    ei[1][e] = edges.center[static_cast<size_t>(e)];
  }
  check_edge_indices(edge_index, n);
  auto cell_offsets = torch::empty({nedges, 3}, opts_f);
  store_rows(cell_offsets, edges.offsets.data(), nedges, 3);
  auto charge = torch::tensor({static_cast<int64_t>(m_config.charge)}, opts_l);
  auto spin = torch::tensor({static_cast<int64_t>(m_config.spin)}, opts_l);
  auto batch = torch::zeros({n}, opts_l);
  auto natoms = torch::tensor({n}, opts_l);

  if (m_impl->device != torch::kCPU) {
    pos = pos.to(m_impl->device);
    atomic_numbers = atomic_numbers.to(m_impl->device);
    cell = cell.to(m_impl->device);
    pbc = pbc.to(m_impl->device);
    edge_index = edge_index.to(m_impl->device);
    cell_offsets = cell_offsets.to(m_impl->device);
    charge = charge.to(m_impl->device);
    spin = spin.to(m_impl->device);
    batch = batch.to(m_impl->device);
    natoms = natoms.to(m_impl->device);
  }

  std::vector<torch::Tensor> inputs = {pos,        atomic_numbers, cell,   pbc,
                                       edge_index, cell_offsets,   charge, spin,
                                       batch,      natoms};

  std::lock_guard<std::mutex> lock(m_impl->mutex);
  ensureLoaded();
  c10::InferenceMode guard;
  auto outputs = m_impl->loader->run(inputs);
  if (outputs.size() < 2)
    throw std::runtime_error("UmaPot: AOTI package returned <2 tensors");

  const double energy = outputs[0].to(torch::kFloat64).cpu().item<double>();
  auto forces = outputs[1].to(torch::kFloat64).cpu().contiguous();
  if (forces.numel() != n * 3)
    throw std::runtime_error("UmaPot: force tensor shape mismatch");
  const double *fp = forces.data_ptr<double>();
  for (int64_t i = 0; i < n * 3; ++i)
    out->F[i] = fp[i];
  out->energy = energy;
  out->variance = 0.0;
}

void UmaPot::forceBatchImpl(const ForceBatch &batch) const {
  // A band evaluation: every system shares this pot's composition, so
  // one AOTI call covers a chunk of up to batch_max geometries. The
  // per-system fallback covers everything else (no band contract in
  // the package, mixed compositions, single systems).
  auto fallback = [&](size_t from, size_t to) {
    for (size_t i = from; i < to; ++i)
      forceImpl(batch.in[i], &batch.out[i]);
  };
  if (batch.nSystems == 0)
    return;
  {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    ensureLoaded();
  }
  const int64_t bmax = m_impl->batch_max;
  if (bmax <= 1) {
    // Single-system package: the plain graph is the only one there is.
    fallback(0, batch.nSystems);
    return;
  }
  const size_t n0 = batch.in[0].nAtoms;
  bool uniform = true;
  for (size_t i = 0; uniform && i < batch.nSystems; ++i) {
    uniform = batch.in[i].nAtoms == n0;
    for (size_t a = 0; uniform && a < n0; ++a)
      uniform = batch.in[i].atmnrs[a] == batch.in[0].atmnrs[a];
  }
  if (!uniform) {
    throw std::runtime_error(
        "UmaPot: band package requires one composition per batch");
  }
  checkContract(batch.in[0]);

  const auto n = static_cast<int64_t>(n0);
  const bool molecular = m_impl->molecular_box > 0.0;
  const double L = m_impl->molecular_box;

  for (size_t start = 0; start < batch.nSystems;
       start += static_cast<size_t>(bmax)) {
    const size_t stop =
        std::min(batch.nSystems, start + static_cast<size_t>(bmax));
    const auto B_real = static_cast<int64_t>(stop - start);
    // The band graph is traced at exactly batch_max systems (static
    // shapes; merge_mole freezes per-atom buffers at the traced
    // total), so every chunk pads to that size by repeating the last
    // system and the padded results are discarded.
    const int64_t B = bmax;
    auto in_at = [&](int64_t b) -> const ForceInput & {
      const int64_t src = b < B_real ? b : B_real - 1;
      return batch.in[start + static_cast<size_t>(src)];
    };

    // Per-system recenter + edges; edge indices offset into the
    // concatenated atom range.
    std::vector<std::vector<double>> sys_pos(static_cast<size_t>(B));
    std::vector<std::array<double, 9>> sys_box(static_cast<size_t>(B));
    std::vector<EdgeList> sys_edges(static_cast<size_t>(B));
    int64_t nedges_total = 0;
    for (int64_t b = 0; b < B; ++b) {
      const ForceInput &in = in_at(b);
      auto &pos_b = sys_pos[static_cast<size_t>(b)];
      auto &box_b = sys_box[static_cast<size_t>(b)];
      pos_b.assign(in.pos, in.pos + 3 * n);
      if (molecular) {
        box_b = uma::molecularFrame(pos_b, L);
      } else {
        std::copy(in.box, in.box + 9, box_b.begin());
      }
      const ForceInput local{static_cast<size_t>(n), pos_b.data(), in.atmnrs,
                             box_b.data()};
      sys_edges[static_cast<size_t>(b)] = build_system_edges(
          local, m_impl->cutoff, m_impl->model_cutoff, m_impl->molecular_box,
          m_impl->max_neighbors, m_impl->contract, B);
      nedges_total += static_cast<int64_t>(
          sys_edges[static_cast<size_t>(b)].neighbor.size());
    }
    if (nedges_total == 0)
      throw std::runtime_error("UmaPot: vesin produced no edges (batch)");
    std::vector<int64_t> per_system(static_cast<size_t>(B));
    for (int64_t b = 0; b < B; ++b) {
      per_system[static_cast<size_t>(b)] = static_cast<int64_t>(
          sys_edges[static_cast<size_t>(b)].neighbor.size());
      if (molecular)
        reject_periodic_images(sys_edges[static_cast<size_t>(b)].offsets);
    }
    // Same edge contract as a single system, including a molecular
    // band. forceImpl sends batch_max > 1 here before any other check.
    apply_edge_contract(m_impl->contract, molecular, n, B, per_system.data(),
                        nedges_total);

    auto opts_f =
        torch::TensorOptions().dtype(m_impl->dtype).device(torch::kCPU);
    auto opts_l =
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);

    auto pos = torch::empty({B * n, 3}, opts_f);
    auto atomic_numbers = torch::empty({B * n}, opts_l);
    auto cell = torch::empty({B, 3, 3}, opts_f);
    auto pbc = torch::ones({B, 3}, torch::TensorOptions().dtype(torch::kBool));
    auto edge_index = torch::empty({2, nedges_total}, opts_l);
    auto cell_offsets = torch::empty({nedges_total, 3}, opts_f);
    auto charge =
        torch::full({B}, static_cast<int64_t>(m_config.charge), opts_l);
    auto spin = torch::full({B}, static_cast<int64_t>(m_config.spin), opts_l);
    auto batch_vec = torch::empty({B * n}, opts_l);
    auto natoms = torch::full({B}, n, opts_l);
    auto nedges_t = torch::empty({B}, opts_l);
    {
      auto ne_a = nedges_t.accessor<int64_t, 1>();
      for (int64_t b = 0; b < B; ++b)
        ne_a[b] = static_cast<int64_t>(
            sys_edges[static_cast<size_t>(b)].neighbor.size());
    }

    {
      auto z_a = atomic_numbers.accessor<int64_t, 1>();
      auto ei = edge_index.accessor<int64_t, 2>();
      auto bv = batch_vec.accessor<int64_t, 1>();
      int64_t e_at = 0;
      for (int64_t b = 0; b < B; ++b) {
        const ForceInput &in = in_at(b);
        const auto &pos_b = sys_pos[static_cast<size_t>(b)];
        store_rows(pos.narrow(0, b * n, n), pos_b.data(), n, 3);
        for (int64_t i = 0; i < n; ++i) {
          z_a[b * n + i] = in.atmnrs[i];
          bv[b * n + i] = b;
        }
        store_rows(cell[b].view({1, 9}), sys_box[static_cast<size_t>(b)].data(),
                   1, 9);
        const auto &edges = sys_edges[static_cast<size_t>(b)];
        const auto ne = static_cast<int64_t>(edges.neighbor.size());
        for (int64_t e = 0; e < ne; ++e) {
          ei[0][e_at + e] = edges.neighbor[static_cast<size_t>(e)] + b * n;
          ei[1][e_at + e] = edges.center[static_cast<size_t>(e)] + b * n;
        }
        if (ne > 0)
          store_rows(cell_offsets.narrow(0, e_at, ne), edges.offsets.data(), ne,
                     3);
        e_at += ne;
      }
    }
    check_edge_indices(edge_index, B * n);
    auto cell_offsets_f = cell_offsets;

    if (m_impl->device != torch::kCPU) {
      pos = pos.to(m_impl->device);
      atomic_numbers = atomic_numbers.to(m_impl->device);
      cell = cell.to(m_impl->device);
      pbc = pbc.to(m_impl->device);
      edge_index = edge_index.to(m_impl->device);
      cell_offsets_f = cell_offsets_f.to(m_impl->device);
      charge = charge.to(m_impl->device);
      spin = spin.to(m_impl->device);
      batch_vec = batch_vec.to(m_impl->device);
      natoms = natoms.to(m_impl->device);
      nedges_t = nedges_t.to(m_impl->device);
    }

    std::vector<torch::Tensor> inputs = {
        pos,    atomic_numbers, cell,      pbc,    edge_index, cell_offsets_f,
        charge, spin,           batch_vec, natoms, nedges_t};

    std::lock_guard<std::mutex> lock(m_impl->mutex);
    c10::InferenceMode guard;
    auto outputs = m_impl->loader->run(inputs);
    if (outputs.size() < 2)
      throw std::runtime_error("UmaPot: AOTI package returned <2 tensors");
    auto energies = outputs[0].to(torch::kFloat64).cpu().contiguous();
    auto forces = outputs[1].to(torch::kFloat64).cpu().contiguous();
    if (energies.numel() != B || forces.numel() != B * n * 3)
      throw std::runtime_error("UmaPot: batch output shape mismatch");
    const double *ep = energies.data_ptr<double>();
    const double *fp = forces.data_ptr<double>();
    for (int64_t b = 0; b < B_real; ++b) {
      ForceOut &o = batch.out[start + static_cast<size_t>(b)];
      o.energy = ep[b];
      o.variance = 0.0;
      for (int64_t i = 0; i < n * 3; ++i)
        o.F[i] = fp[b * n * 3 + i];
    }
  }
}

const char *UmaPot::posDtypeName() const {
  std::lock_guard<std::mutex> lock(m_impl->mutex);
  ensureLoaded();
  return m_impl->pos_dtype_name.c_str();
}

const char *UmaPot::modelDtypeName() const {
  std::lock_guard<std::mutex> lock(m_impl->mutex);
  ensureLoaded();
  return m_impl->model_dtype_name.c_str();
}

} // namespace rgpot
