// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error;
std::int64_t g_drop_count = 0;
std::mutex g_drop_count_mutex;

void set_error(const std::string& text) { g_last_error = text; }

enum StatusCode : int32_t {
  RGPOT_SUCCESS = 0,
  RGPOT_INVALID_PARAMETER = 1,
  RGPOT_INTERNAL_ERROR = 2,
};

enum Format : int32_t {
  AUTO = 0,
  FUNCFL = 1,
  SETFL = 2,
  FINNIS_SINCLAIR = 3,
};

constexpr double kHartreeBohr = 27.2 * 0.529;

struct EamTable {
  std::size_t nrho = 0;
  double drho = 0.0;
  std::size_t nr = 0;
  double dr = 0.0;
  double cutoff = 0.0;
  double rhomax = 0.0;
  std::vector<int32_t> z;
  std::vector<std::vector<double>> frho;
  std::vector<std::vector<double>> rhor;
  std::vector<std::vector<double>> z2r;
};

struct Grid {
  std::size_t nrho = 0;
  double drho = 0.0;
  std::size_t nr = 0;
  double dr = 0.0;
  double cutoff = 0.0;
};

enum class Layout { Funcfl, Setfl, FinnisSinclair };

class Scanner {
 public:
  explicit Scanner(const std::string& text) : rest_(text.c_str()) {}

  void skip_line() {
    const char* nl = std::strchr(rest_, '\n');
    rest_ = nl == nullptr ? rest_ + std::strlen(rest_) : nl + 1;
  }

  template <typename T>
  T number(const char* what) {
    std::istringstream token(next_token());
    T out{};
    if (!(token >> out)) {
      throw std::invalid_argument(std::string("EAM table: unreadable ") + what);
    }
    return out;
  }

  int32_t element_line() {
    const int32_t z = number<int32_t>("element line");
    (void)number<double>("element line");
    skip_line();
    return z;
  }

  Grid grid_line() {
    const auto nrho = number<int64_t>("grid line");
    const double drho = number<double>("grid line");
    const auto nr = number<int64_t>("grid line");
    const double dr = number<double>("grid line");
    const double cutoff = number<double>("grid line");
    skip_line();
    const bool positive = drho > 0.0 && dr > 0.0 && cutoff > 0.0 && std::isfinite(drho) &&
                          std::isfinite(dr) && std::isfinite(cutoff);
    if (nrho < 5 || nr < 5 || !positive) {
      throw std::invalid_argument("EAM table: unreadable grid line");
    }
    return {static_cast<std::size_t>(nrho), drho, static_cast<std::size_t>(nr), dr, cutoff};
  }

  std::vector<double> block(std::size_t n) {
    std::vector<double> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double value = number<double>("array");
      if (!std::isfinite(value)) {
        throw std::invalid_argument("EAM table: short or unreadable array");
      }
      out.push_back(value);
    }
    return out;
  }

 private:
  std::string next_token() {
    while (*rest_ != '\0' && std::isspace(static_cast<unsigned char>(*rest_))) {
      ++rest_;
    }
    if (*rest_ == '\0') {
      throw std::invalid_argument("EAM table: short or unreadable array");
    }
    const char* begin = rest_;
    while (*rest_ != '\0' && !std::isspace(static_cast<unsigned char>(*rest_))) {
      ++rest_;
    }
    return std::string(begin, static_cast<std::size_t>(rest_ - begin));
  }

  const char* rest_;
};

std::vector<double> resample(const std::vector<double>& f, double fd, std::size_t n, double d) {
  const double sixth = 1.0 / 6.0;
  const auto nf = static_cast<int64_t>(f.size());
  std::vector<double> out;
  out.reserve(n);
  for (std::size_t m = 1; m <= n; ++m) {
    double p = static_cast<double>(m - 1) * d / fd + 1.0;
    const auto k = std::max<int64_t>(2, std::min<int64_t>(static_cast<int64_t>(p), nf - 2));
    p = std::min(p - static_cast<double>(k), 2.0);
    const double c1 = -sixth * p * (p - 1.0) * (p - 2.0);
    const double c2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    const double c3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    const double c4 = sixth * p * (p * p - 1.0);
    const auto ku = static_cast<std::size_t>(k);
    out.push_back(c1 * f[ku - 2] + c2 * f[ku - 1] + c3 * f[ku] + c4 * f[ku + 1]);
  }
  return out;
}

std::vector<double> spline(const std::vector<double>& f, double delta) {
  const auto n = f.size();
  std::vector<double> s(7 * n, 0.0);
  const auto at = [](std::size_t m, std::size_t k) { return 7 * (m - 1) + k; };
  for (std::size_t m = 1; m <= n; ++m) {
    s[at(m, 6)] = f[m - 1];
  }
  s[at(1, 5)] = s[at(2, 6)] - s[at(1, 6)];
  s[at(2, 5)] = 0.5 * (s[at(3, 6)] - s[at(1, 6)]);
  s[at(n - 1, 5)] = 0.5 * (s[at(n, 6)] - s[at(n - 2, 6)]);
  s[at(n, 5)] = s[at(n, 6)] - s[at(n - 1, 6)];
  for (std::size_t m = 3; m <= n - 2; ++m) {
    s[at(m, 5)] =
        ((s[at(m - 2, 6)] - s[at(m + 2, 6)]) + 8.0 * (s[at(m + 1, 6)] - s[at(m - 1, 6)])) /
        12.0;
  }
  for (std::size_t m = 1; m <= n - 1; ++m) {
    s[at(m, 4)] = 3.0 * (s[at(m + 1, 6)] - s[at(m, 6)]) - 2.0 * s[at(m, 5)] - s[at(m + 1, 5)];
    s[at(m, 3)] = s[at(m, 5)] + s[at(m + 1, 5)] - 2.0 * (s[at(m + 1, 6)] - s[at(m, 6)]);
  }
  s[at(n, 4)] = 0.0;
  s[at(n, 3)] = 0.0;
  for (std::size_t m = 1; m <= n; ++m) {
    s[at(m, 2)] = s[at(m, 5)] / delta;
    s[at(m, 1)] = 2.0 * s[at(m, 4)] / delta;
    s[at(m, 0)] = 3.0 * s[at(m, 3)] / delta;
  }
  return s;
}

struct Place {
  std::size_t offset = 0;
  double p = 0.0;
};

Place place(double x, double rdx, std::size_t n) {
  const double q = x * rdx + 1.0;
  const auto m =
      std::max<int64_t>(1, std::min<int64_t>(static_cast<int64_t>(q), static_cast<int64_t>(n) - 1));
  return {7 * (static_cast<std::size_t>(m) - 1), std::min(q - static_cast<double>(m), 1.0)};
}

struct Knot {
  const double* c = nullptr;
  double p = 0.0;
  Knot(const std::vector<double>& table, Place at) : c(table.data() + at.offset), p(at.p) {}
  [[nodiscard]] double value() const { return ((c[3] * p + c[4]) * p + c[5]) * p + c[6]; }
  [[nodiscard]] double slope() const { return (c[0] * p + c[1]) * p + c[2]; }
};

Layout layout_of(Format format, const std::string& path_hint) {
  if (format == FUNCFL) {
    return Layout::Funcfl;
  }
  if (format == SETFL) {
    return Layout::Setfl;
  }
  if (format == FINNIS_SINCLAIR) {
    return Layout::FinnisSinclair;
  }
  if (format != AUTO) {
    throw std::invalid_argument("EAM table: unknown format code");
  }
  std::string lower = path_hint;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower.size() >= 10 && lower.ends_with(".eam.alloy")) {
    return Layout::Setfl;
  }
  if (lower.size() >= 6 && lower.ends_with(".setfl")) {
    return Layout::Setfl;
  }
  if (lower.size() >= 7 && lower.ends_with(".eam.fs")) {
    return Layout::FinnisSinclair;
  }
  if (lower.size() >= 4 && lower.ends_with(".eam")) {
    return Layout::Funcfl;
  }
  throw std::invalid_argument("EAM table: cannot tell table format from file suffix");
}

EamTable parse_table(const std::string& text, Layout layout) {
  Scanner in(text);
  EamTable table;
  if (layout == Layout::Funcfl) {
    in.skip_line();
    table.z.push_back(in.element_line());
    const Grid g = in.grid_line();
    auto f = in.block(g.nrho);
    auto zr = in.block(g.nr);
    auto rho = in.block(g.nr);
    table.rhomax = static_cast<double>(g.nrho - 1) * g.drho;
    const auto nrho = static_cast<std::size_t>(std::llround(table.rhomax / g.drho));
    const auto nr = static_cast<std::size_t>(
        std::llround(static_cast<double>(g.nr - 1) * g.dr / g.dr));
    auto z2 = resample(zr, g.dr, nr, g.dr);
    for (double& v : z2) {
      v = kHartreeBohr * v * v;
    }
    table.nrho = nrho;
    table.drho = g.drho;
    table.nr = nr;
    table.dr = g.dr;
    table.cutoff = g.cutoff;
    table.frho = {resample(f, g.drho, nrho, g.drho)};
    table.rhor = {resample(rho, g.dr, nr, g.dr)};
    table.z2r = {std::move(z2)};
    return table;
  }

  for (int i = 0; i < 3; ++i) {
    in.skip_line();
  }
  const auto nel_i = in.number<int64_t>("element count");
  in.skip_line();
  if (nel_i < 1) {
    throw std::invalid_argument("EAM table: unreadable element count");
  }
  const auto nel = static_cast<std::size_t>(nel_i);
  const Grid g = in.grid_line();
  const auto per_element = layout == Layout::FinnisSinclair ? nel : std::size_t{1};
  std::vector<std::vector<double>> dens;
  for (std::size_t b = 0; b < nel; ++b) {
    table.z.push_back(in.element_line());
    table.frho.push_back(in.block(g.nrho));
    for (std::size_t a = 0; a < per_element; ++a) {
      dens.push_back(in.block(g.nr));
    }
  }
  for (std::size_t a = 0; a < nel; ++a) {
    for (std::size_t b = 0; b <= a; ++b) {
      table.z2r.push_back(in.block(g.nr));
    }
  }
  for (std::size_t b = 0; b < nel; ++b) {
    for (std::size_t a = 0; a < nel; ++a) {
      table.rhor.push_back(dens[b * per_element + (a % per_element)]);
    }
  }
  table.nrho = g.nrho;
  table.drho = g.drho;
  table.nr = g.nr;
  table.dr = g.dr;
  table.cutoff = g.cutoff;
  table.rhomax = static_cast<double>(g.nrho - 1) * g.drho;
  return table;
}

class EAMTablePot {
 public:
  EAMTablePot(EamTable table, bool periodic, bool stress, double skin)
      : table_(std::move(table)), periodic_(periodic), stress_(stress), skin_(skin) {
    if (!std::isfinite(skin_) || skin_ < 0.0) {
      throw std::invalid_argument("EAM table: skin must be finite and non-negative");
    }
    const auto n = table_.z.size();
    if (n == 0 || table_.frho.size() != n || table_.rhor.size() != n * n ||
        table_.z2r.size() != n * (n + 1) / 2) {
      throw std::invalid_argument("EAM table: array count does not match");
    }
    for (const auto& array : table_.frho) {
      if (array.size() != table_.nrho || array.size() < 4) {
        throw std::invalid_argument("EAM table: array length does not match");
      }
      frho_.push_back(spline(array, table_.drho));
    }
    for (const auto& array : table_.rhor) {
      if (array.size() != table_.nr || array.size() < 4) {
        throw std::invalid_argument("EAM table: array length does not match");
      }
      rhor_.push_back(spline(array, table_.dr));
    }
    for (const auto& array : table_.z2r) {
      if (array.size() != table_.nr || array.size() < 4) {
        throw std::invalid_argument("EAM table: array length does not match");
      }
      z2r_.push_back(spline(array, table_.dr));
    }
  }

  void force(std::size_t n_atoms, const double* positions, const int32_t* numbers,
             const double* cell, double* forces, double* energy) const {
    if (positions == nullptr || numbers == nullptr || forces == nullptr || energy == nullptr) {
      throw std::invalid_argument("EAM table: null force argument");
    }
    for (std::size_t i = 0; i < 3 * n_atoms; ++i) {
      if (!std::isfinite(positions[i])) {
        throw std::invalid_argument("EAM table: non-finite coordinate");
      }
      forces[i] = 0.0;
    }
    std::vector<std::size_t> element(n_atoms, 0);
    for (std::size_t i = 0; i < n_atoms; ++i) {
      bool found = false;
      for (std::size_t a = 0; a < table_.z.size(); ++a) {
        if (table_.z[a] == numbers[i]) {
          element[i] = a;
          found = true;
          break;
        }
      }
      if (!found) {
        throw std::invalid_argument("EAM table: no element for atomic number " +
                                    std::to_string(numbers[i]));
      }
    }
    std::array<double, 9> cell_mat{};
    if (periodic_) {
      if (cell == nullptr) {
        throw std::invalid_argument("EAM table: periodic evaluation needs a cell");
      }
      for (std::size_t i = 0; i < 9; ++i) {
        cell_mat[i] = cell[i];
        if (!std::isfinite(cell_mat[i])) {
          throw std::invalid_argument("EAM table: non-finite cell");
        }
      }
    }

    std::array<double, 9> inverse{};
    if (periodic_) {
      inverse = cell_inverse(cell_mat);
    }

    std::vector<double> rho(n_atoms, 0.0);
    struct Pair {
      std::size_t i;
      std::size_t j;
      std::array<double, 3> d;
      double r;
    };
    std::vector<Pair> pairs;
    const double cut2 = table_.cutoff * table_.cutoff;
    for (std::size_t i = 0; i < n_atoms; ++i) {
      for (std::size_t j = i + 1; j < n_atoms; ++j) {
        std::array<double, 3> d{
            positions[3 * i] - positions[3 * j], positions[3 * i + 1] - positions[3 * j + 1],
            positions[3 * i + 2] - positions[3 * j + 2]};
        if (periodic_) {
          d = nearest_image(d, cell_mat, inverse);
        }
        const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if (r2 > 0.0 && r2 < cut2) {
          pairs.push_back(Pair{i, j, d, std::sqrt(r2)});
        }
      }
    }

    for (const auto& pair : pairs) {
      const auto [at_i, at_j, _] = couple(element[pair.i], element[pair.j]);
      const auto at = place(pair.r, 1.0 / table_.dr, table_.nr);
      rho[pair.i] += Knot(*at_i, at).value();
      rho[pair.j] += Knot(*at_j, at).value();
    }

    std::vector<double> fprime(n_atoms, 0.0);
    double total_energy = 0.0;
    for (std::size_t i = 0; i < n_atoms; ++i) {
      auto [embed, slope] = embedding(element[i], rho[i]);
      total_energy += embed;
      fprime[i] = slope;
    }

    for (const auto& pair : pairs) {
      const auto [at_i, at_j, z2] = couple(element[pair.i], element[pair.j]);
      const auto at = place(pair.r, 1.0 / table_.dr, table_.nr);
      const double rhoip = Knot(*at_j, at).slope();
      const double rhojp = Knot(*at_i, at).slope();
      const Knot z2k(*z2, at);
      const double recip = 1.0 / pair.r;
      const double phi = z2k.value() * recip;
      const double phip = z2k.slope() * recip - phi * recip;
      const double fscale = -(fprime[pair.i] * rhojp + fprime[pair.j] * rhoip + phip) * recip;
      total_energy += phi;
      for (std::size_t k = 0; k < 3; ++k) {
        forces[3 * pair.i + k] += fscale * pair.d[k];
        forces[3 * pair.j + k] -= fscale * pair.d[k];
      }
    }
    if (!std::isfinite(total_energy)) {
      throw std::invalid_argument("EAM table: non-finite energy");
    }
    for (std::size_t i = 0; i < 3 * n_atoms; ++i) {
      if (!std::isfinite(forces[i])) {
        throw std::invalid_argument("EAM table: non-finite force");
      }
    }
    *energy = total_energy;
    (void)stress_;
    (void)skin_;
  }

 private:
  std::tuple<const std::vector<double>*, const std::vector<double>*, const std::vector<double>*>
  couple(std::size_t a, std::size_t b) const {
    const auto n = table_.z.size();
    const auto hi = std::max(a, b);
    return {&rhor_[b * n + a], &rhor_[a * n + b], &z2r_[hi * (hi + 1) / 2 + std::min(a, b)]};
  }

  std::pair<double, double> embedding(std::size_t a, double rho) const {
    const auto at = place(rho, 1.0 / table_.drho, table_.nrho);
    Knot k(frho_[a], at);
    const double slope = k.slope();
    double value = k.value();
    if (rho > table_.rhomax) {
      value += slope * (rho - table_.rhomax);
    }
    return {value, slope};
  }

  static std::array<double, 3> nearest_image(const std::array<double, 3>& d,
                                             const std::array<double, 9>& cell,
                                             const std::array<double, 9>& inverse) {
    std::array<double, 3> s{};
    for (std::size_t k = 0; k < 3; ++k) {
      const double frac = d[0] * inverse[k] + d[1] * inverse[3 + k] + d[2] * inverse[6 + k];
      s[k] = frac - std::round(frac);
    }
    std::array<double, 3> out{};
    for (std::size_t k = 0; k < 3; ++k) {
      out[k] = s[0] * cell[k] + s[1] * cell[3 + k] + s[2] * cell[6 + k];
    }
    return out;
  }

  std::array<double, 9> cell_inverse(const std::array<double, 9>& cell) const {
    auto row = [&](std::size_t k) { return std::array<double, 3>{cell[3 * k], cell[3 * k + 1], cell[3 * k + 2]}; };
    const auto cross = [](const std::array<double, 3>& u, const std::array<double, 3>& v) {
      return std::array<double, 3>{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                                   u[0] * v[1] - u[1] * v[0]};
    };
    std::array<std::array<double, 3>, 3> normals{cross(row(1), row(2)), cross(row(2), row(0)),
                                                  cross(row(0), row(1))};
    const auto a = row(0);
    const double det = a[0] * normals[0][0] + a[1] * normals[0][1] + a[2] * normals[0][2];
    const double volume = std::abs(det);
    if (!(volume > 0.0) || !std::isfinite(volume)) {
      throw std::invalid_argument("EAM table: periodic evaluation needs a cell");
    }
    for (const auto& n : normals) {
      const double area = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      if (!(volume / area > 2.0 * table_.cutoff)) {
        throw std::invalid_argument("EAM table: cutoff exceeds half a face spacing");
      }
    }
    std::array<double, 9> inverse{};
    for (std::size_t m = 0; m < 3; ++m) {
      for (std::size_t k = 0; k < 3; ++k) {
        inverse[3 * m + k] = normals[k][m] / det;
      }
    }
    return inverse;
  }

  EamTable table_;
  std::vector<std::vector<double>> frho_;
  std::vector<std::vector<double>> rhor_;
  std::vector<std::vector<double>> z2r_;
  bool periodic_ = true;
  bool stress_ = false;
  double skin_ = 1.0;
};

std::string read_file(const char* path) {
  if (path == nullptr) {
    throw std::invalid_argument("EAM table: path is NULL");
  }
  std::ifstream in(path);
  if (!in) {
    throw std::invalid_argument(std::string("EAM table: cannot open ") + path);
  }
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

int to_status(const std::exception& e) {
  const std::string what = e.what();
  set_error(what);
  return what.find("non-finite") != std::string::npos ? RGPOT_INTERNAL_ERROR
                                                       : RGPOT_INVALID_PARAMETER;
}

}  // namespace

extern "C" {

const char* rgpot_core_eam_cpp_last_error() { return g_last_error.c_str(); }

void* rgpot_core_eam_cpp_new_from_file(const char* path, int32_t format, bool periodic,
                                       bool stress, double skin) {
  try {
    const auto layout = layout_of(static_cast<Format>(format), path == nullptr ? "" : path);
    auto table = parse_table(read_file(path), layout);
    return new EAMTablePot(std::move(table), periodic, stress, skin);
  } catch (const std::exception& e) {
    set_error(e.what());
    return nullptr;
  } catch (...) {
    set_error("EAM table: unknown exception while constructing EAMTablePot");
    return nullptr;
  }
}

void* rgpot_core_eam_cpp_new_from_text(const char* text, int32_t format, bool periodic,
                                       bool stress, double skin) {
  try {
    if (text == nullptr) {
      throw std::invalid_argument("EAM table: text is NULL");
    }
    auto table = parse_table(std::string(text), layout_of(static_cast<Format>(format), "table.eam.fs"));
    return new EAMTablePot(std::move(table), periodic, stress, skin);
  } catch (const std::exception& e) {
    set_error(e.what());
    return nullptr;
  } catch (...) {
    set_error("EAM table: unknown exception while constructing EAMTablePot");
    return nullptr;
  }
}

void rgpot_core_eam_cpp_free(void* handle) {
  if (handle != nullptr) {
    delete static_cast<EAMTablePot*>(handle);
    std::lock_guard<std::mutex> hold(g_drop_count_mutex);
    ++g_drop_count;
  }
}

int32_t rgpot_core_eam_cpp_force(void* handle, std::size_t n_atoms, const double* positions,
                                 const int32_t* atomic_numbers, const double* cell,
                                 double* forces, double* energy) {
  try {
    if (handle == nullptr) {
      throw std::invalid_argument("EAM table: null potential handle");
    }
    static_cast<EAMTablePot*>(handle)->force(n_atoms, positions, atomic_numbers, cell, forces,
                                             energy);
    return RGPOT_SUCCESS;
  } catch (const std::exception& e) {
    return to_status(e);
  } catch (...) {
    set_error("EAM table: unknown exception while evaluating EAMTablePot");
    return RGPOT_INTERNAL_ERROR;
  }
}

int64_t rgpot_core_eam_cpp_drop_count() {
  std::lock_guard<std::mutex> hold(g_drop_count_mutex);
  return g_drop_count;
}

void rgpot_core_eam_cpp_reset_drop_count() {
  std::lock_guard<std::mutex> hold(g_drop_count_mutex);
  g_drop_count = 0;
}

}  // extern "C"
