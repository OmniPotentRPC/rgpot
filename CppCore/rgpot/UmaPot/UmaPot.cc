// MIT License — UmaPot: vesin neighbor list + AOTInductor .pt2

#include "rgpot/UmaPot/UmaPot.hpp"

#include "rgpot/UmaPot/aoti_execstack.hpp"
#include "rgpot/MetatomicPot/vesin_compat.hpp"
#include "vesin.h"

#include <array>
#include <vector>

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <mutex>
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
  std::vector<float> offsets; // nedges * 3
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
  const int status = vesin_neighbors(
      reinterpret_cast<const double(*)[3]>(in.pos), in.nAtoms,
      reinterpret_cast<const double(*)[3]>(in.box), pbc, cpu, options, &nl,
      &err);
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
    float s0{0};
    float s1{0};
    float s2{0};
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
    c.s0 = static_cast<float>(nl.shifts[k][0]);
    c.s1 = static_cast<float>(nl.shifts[k][1]);
    c.s2 = static_cast<float>(nl.shifts[k][2]);
    c.dist = dist;
    keep.push_back(c);
  }
  vesin_free(&nl);

  // Canonical edge order: the neighbor list may be assembled by several
  // threads, so its pair order is not a function of the input. The model's
  // float32 message-passing sums follow the edge order, so a run is
  // reproducible only if this order is. Total order on (center, distance,
  // neighbor, shift).
  std::sort(keep.begin(), keep.end(), [](const Cand &a, const Cand &b) {
    if (a.c != b.c) return a.c < b.c;
    if (a.dist != b.dist) return a.dist < b.dist;
    if (a.n != b.n) return a.n < b.n;
    if (a.s0 != b.s0) return a.s0 < b.s0;
    if (a.s1 != b.s1) return a.s1 < b.s1;
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

} // namespace

struct UmaPot::Impl {
  torch::Device device{torch::kCPU};
  torch::ScalarType dtype{torch::kFloat32};
  double cutoff{6.0};
  double molecular_box{0.0};
  int max_neighbors{300};
  int64_t batch_max{0};
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
  m_impl->molecular_box = num("molecular_box", m_impl->molecular_box);
  m_impl->max_neighbors = static_cast<int>(
      num("max_neighbors", static_cast<double>(m_impl->max_neighbors)));
  m_impl->batch_max = static_cast<int64_t>(num("batch_max", 0.0));
  const auto dt = meta.find("pos_dtype");
  if (dt != meta.end() && dt->second == "float64")
    m_impl->dtype = torch::kFloat64;
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
  if (m_impl->batch_max > 1) {
    // A band package carries only the batched (B >= 2) graph; a lone
    // system rides the batch path, which pads it to two.
    ForceBatch one{.nSystems = 1, .in = &in, .out = out};
    forceBatchImpl(one);
    return;
  }

  // Under the molecular-box convention the caller's cell is replaced by
  // the sidecar's cube and positions re-center into it. Energies and
  // forces are translation invariant, so only the graph changes.
  const bool molecular = m_impl->molecular_box > 0.0;
  std::vector<double> mol_pos;
  std::array<double, 9> mol_box{};
  if (molecular) {
    const double L = m_impl->molecular_box;
    mol_pos.assign(in.pos, in.pos + 3 * in.nAtoms);
    double c[3] = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < in.nAtoms; ++i)
      for (int d = 0; d < 3; ++d)
        c[d] += mol_pos[3 * i + d];
    for (int d = 0; d < 3; ++d)
      c[d] /= static_cast<double>(in.nAtoms);
    for (size_t i = 0; i < in.nAtoms; ++i)
      for (int d = 0; d < 3; ++d)
        mol_pos[3 * i + d] += 0.5 * L - c[d];
    mol_box = {L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L};
  }
  const ForceInput local{in.nAtoms, molecular ? mol_pos.data() : in.pos,
                         in.atmnrs, molecular ? mol_box.data() : in.box};

  const EdgeList edges =
      vesin_fairchem_edges(local, m_impl->cutoff, m_impl->max_neighbors);
  const auto n = static_cast<int64_t>(local.nAtoms);
  const auto nedges = static_cast<int64_t>(edges.neighbor.size());
  if (nedges == 0)
    throw std::runtime_error("UmaPot: vesin produced no edges");

  auto opts_f = torch::TensorOptions()
                    .dtype(m_impl->dtype)
                    .device(torch::kCPU);
  auto opts_l = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);

  auto pos = torch::empty({n, 3}, torch::dtype(torch::kFloat64));
  auto pos_a = pos.accessor<double, 2>();
  for (int64_t i = 0; i < n; ++i) {
    pos_a[i][0] = local.pos[3 * i];
    pos_a[i][1] = local.pos[3 * i + 1];
    pos_a[i][2] = local.pos[3 * i + 2];
  }
  pos = pos.to(opts_f);

  auto atomic_numbers = torch::empty({n}, opts_l);
  auto z_a = atomic_numbers.accessor<int64_t, 1>();
  for (int64_t i = 0; i < n; ++i)
    z_a[i] = local.atmnrs[i];

  auto cell = torch::empty({1, 3, 3}, torch::dtype(torch::kFloat64));
  auto cell_a = cell.accessor<double, 3>();
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      cell_a[0][r][c] = local.box[3 * r + c];
  cell = cell.to(opts_f);

  auto pbc = torch::tensor({{true, true, true}},
                           torch::TensorOptions().dtype(torch::kBool));
  auto edge_index = torch::empty({2, nedges}, opts_l);
  auto ei = edge_index.accessor<int64_t, 2>();
  for (int64_t e = 0; e < nedges; ++e) {
    ei[0][e] = edges.neighbor[static_cast<size_t>(e)];
    ei[1][e] = edges.center[static_cast<size_t>(e)];
  }
  auto cell_offsets =
      torch::from_blob(const_cast<float *>(edges.offsets.data()),
                       {nedges, 3}, torch::dtype(torch::kFloat32))
          .clone()
          .to(opts_f);
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

  std::vector<torch::Tensor> inputs = {pos,         atomic_numbers, cell,
                                       pbc,         edge_index,     cell_offsets,
                                       charge,      spin,           batch,
                                       natoms};

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
        double c[3] = {0.0, 0.0, 0.0};
        for (int64_t i = 0; i < n; ++i)
          for (int d = 0; d < 3; ++d)
            c[d] += pos_b[static_cast<size_t>(3 * i + d)];
        for (int d = 0; d < 3; ++d)
          c[d] /= static_cast<double>(n);
        for (int64_t i = 0; i < n; ++i)
          for (int d = 0; d < 3; ++d)
            pos_b[static_cast<size_t>(3 * i + d)] += 0.5 * L - c[d];
        box_b = {L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L};
      } else {
        std::copy(in.box, in.box + 9, box_b.begin());
      }
      const ForceInput local{static_cast<size_t>(n), pos_b.data(), in.atmnrs,
                             box_b.data()};
      sys_edges[static_cast<size_t>(b)] = vesin_fairchem_edges(
          local, m_impl->cutoff, m_impl->max_neighbors);
      nedges_total += static_cast<int64_t>(
          sys_edges[static_cast<size_t>(b)].neighbor.size());
    }
    if (nedges_total == 0)
      throw std::runtime_error("UmaPot: vesin produced no edges (batch)");

    auto opts_f =
        torch::TensorOptions().dtype(m_impl->dtype).device(torch::kCPU);
    auto opts_l =
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);

    auto pos = torch::empty({B * n, 3}, torch::dtype(torch::kFloat64));
    auto atomic_numbers = torch::empty({B * n}, opts_l);
    auto cell = torch::empty({B, 3, 3}, torch::dtype(torch::kFloat64));
    auto pbc = torch::ones({B, 3}, torch::TensorOptions().dtype(torch::kBool));
    auto edge_index = torch::empty({2, nedges_total}, opts_l);
    auto cell_offsets =
        torch::empty({nedges_total, 3}, torch::dtype(torch::kFloat32));
    auto charge = torch::full({B}, static_cast<int64_t>(m_config.charge),
                              opts_l);
    auto spin =
        torch::full({B}, static_cast<int64_t>(m_config.spin), opts_l);
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
      auto pos_a = pos.accessor<double, 2>();
      auto z_a = atomic_numbers.accessor<int64_t, 1>();
      auto cell_a = cell.accessor<double, 3>();
      auto ei = edge_index.accessor<int64_t, 2>();
      auto off = cell_offsets.accessor<float, 2>();
      auto bv = batch_vec.accessor<int64_t, 1>();
      int64_t e_at = 0;
      for (int64_t b = 0; b < B; ++b) {
        const ForceInput &in = in_at(b);
        const auto &pos_b = sys_pos[static_cast<size_t>(b)];
        for (int64_t i = 0; i < n; ++i) {
          pos_a[b * n + i][0] = pos_b[static_cast<size_t>(3 * i)];
          pos_a[b * n + i][1] = pos_b[static_cast<size_t>(3 * i + 1)];
          pos_a[b * n + i][2] = pos_b[static_cast<size_t>(3 * i + 2)];
          z_a[b * n + i] = in.atmnrs[i];
          bv[b * n + i] = b;
        }
        for (int r = 0; r < 3; ++r)
          for (int c = 0; c < 3; ++c)
            cell_a[b][r][c] = sys_box[static_cast<size_t>(b)]
                                     [static_cast<size_t>(3 * r + c)];
        const auto &edges = sys_edges[static_cast<size_t>(b)];
        const auto ne = static_cast<int64_t>(edges.neighbor.size());
        for (int64_t e = 0; e < ne; ++e) {
          ei[0][e_at + e] = edges.neighbor[static_cast<size_t>(e)] + b * n;
          ei[1][e_at + e] = edges.center[static_cast<size_t>(e)] + b * n;
          off[e_at + e][0] = edges.offsets[static_cast<size_t>(3 * e)];
          off[e_at + e][1] = edges.offsets[static_cast<size_t>(3 * e + 1)];
          off[e_at + e][2] = edges.offsets[static_cast<size_t>(3 * e + 2)];
        }
        e_at += ne;
      }
    }
    pos = pos.to(opts_f);
    cell = cell.to(opts_f);
    auto cell_offsets_f = cell_offsets.to(opts_f);

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
        pos,    atomic_numbers, cell,      pbc,    edge_index,
        cell_offsets_f, charge, spin, batch_vec, natoms, nedges_t};

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

} // namespace rgpot
