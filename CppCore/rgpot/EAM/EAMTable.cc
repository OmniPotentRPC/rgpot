// MIT License
// Copyright 2023--present rgpot developers

// clang-format off
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
// clang-format on

#include "rgpot/EAM/EAMTable.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/nlist/PairListCache.hpp"

namespace {

using Table = std::vector<double>;

struct Reader {
  std::ifstream in;
  std::string path;

  [[noreturn]] void fail(const char *what) const {
    throw std::runtime_error("EAM table " + path + ": " + what);
  }

  /// Next line. Unless raw, '#' starts a comment and blank lines are
  /// skipped, as LAMMPS reads past the three title lines.
  std::istringstream line(bool raw = false) {
    std::string s;
    while (std::getline(in, s)) {
      s = raw ? s : s.substr(0, s.find('#'));
      if (raw || s.find_first_not_of(" \t\r") != std::string::npos) {
        return std::istringstream(s);
      }
    }
    fail("unexpected end");
  }

  /// n values from as many lines as they span, stored from index 1.
  Table values(int n) {
    Table v(static_cast<std::size_t>(n) + 1, 0.0);
    for (std::size_t k = 1; k <= v.size() - 1;) {
      std::istringstream ls = line();
      while (k < v.size() && ls >> v[k]) {
        ++k;
      }
      if (k < v.size() && !ls.eof()) {
        fail("bad number");
      }
    }
    return v;
  }
};

/// PairEAM::interpolate: knots 1..n, 0-2 the slope and 3-6 the value.
std::vector<std::array<double, 7>> spline(const Table &f, int n, double h) {
  std::vector<std::array<double, 7>> s(f.size());
  for (std::size_t m = 1; m < s.size(); ++m) {
    s[m][6] = f[m];
  }
  const auto e = static_cast<std::size_t>(n);
  s[1][5] = s[2][6] - s[1][6];
  s[2][5] = 0.5 * (s[3][6] - s[1][6]);
  s[e - 1][5] = 0.5 * (s[e][6] - s[e - 2][6]);
  s[e][5] = s[e][6] - s[e - 1][6];
  for (std::size_t m = 3; m + 2 <= e; ++m) {
    s[m][5] =
        ((s[m - 2][6] - s[m + 2][6]) + 8.0 * (s[m + 1][6] - s[m - 1][6])) /
        12.0;
  }
  for (std::size_t m = 1; m < e; ++m) {
    s[m][4] = 3.0 * (s[m + 1][6] - s[m][6]) - 2.0 * s[m][5] - s[m + 1][5];
    s[m][3] = s[m][5] + s[m + 1][5] - 2.0 * (s[m + 1][6] - s[m][6]);
  }
  for (std::size_t m = 1; m <= e; ++m) {
    s[m][2] = s[m][5] / h;
    s[m][1] = 2.0 * s[m][4] / h;
    s[m][0] = 3.0 * s[m][3] / h;
  }
  return s;
}

double value(const std::array<double, 7> &c, double p) {
  return ((c[3] * p + c[4]) * p + c[5]) * p + c[6];
}

double slope(const std::array<double, 7> &c, double p) {
  return (c[0] * p + c[1]) * p + c[2];
}

} // namespace

namespace rgpot {

EAMTableData read_eam_table(const std::string &path, EAMTableFormat format) {
  Reader rd{std::ifstream(path), path};
  if (!rd.in) {
    rd.fail("cannot open");
  }
  for (int k = 0; k < 3; ++k) {
    rd.line(true);
  }
  EAMTableData d;
  int nel = 0;
  std::istringstream names = rd.line();
  names >> nel;
  int count = 0;
  for (std::string name; names >> name;) {
    ++count;
  }
  if (!names.eof() || count != nel ||
      !(rd.line() >> d.nrho >> d.drho >> d.nr >> d.dr >> d.cut) || nel < 1 ||
      d.nrho < 5 || d.nr < 5 || !(d.drho > 0.0) || !(d.dr > 0.0)) {
    rd.fail("bad element count or grid line");
  }
  const auto n = static_cast<std::size_t>(nel);
  d.frho.resize(n);
  d.rhor.resize(n * n);
  d.z2r.resize(n * n);
  for (std::size_t a = 0; a < n; ++a) {
    int z = 0;
    if (!(rd.line() >> z)) {
      rd.fail("bad element line");
    }
    d.atomic_numbers.push_back(z);
    d.frho[a] = rd.values(d.nrho);
    for (std::size_t b = 0; b < n; ++b) {
      const bool fs = format == EAMTableFormat::FinnisSinclair;
      d.rhor[a * n + b] = (fs || b == 0) ? rd.values(d.nr) : d.rhor[a * n];
    }
  }
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t b = 0; b <= a; ++b) {
      d.z2r[a * n + b] = rd.values(d.nr);
      d.z2r[b * n + a] = d.z2r[a * n + b];
    }
  }
  return d;
}

EAMTablePot::EAMTablePot(const EAMTableConfig &c)
    : Potential(PotType::EAMTable), m_data(read_eam_table(c.path, c.format)),
      m_periodic{c.periodic} {
  if (!c.atomic_numbers.empty()) {
    if (c.atomic_numbers.size() != m_data.atomic_numbers.size()) {
      throw std::invalid_argument(
          "EAM table atomic_numbers needs one entry per file element");
    }
    m_data.atomic_numbers = c.atomic_numbers;
  }
  Fnv1a fp;
  fp.u64(kKernelVersion);
  fp.u64(m_periodic ? 1 : 0);
  fp.f64(m_data.cut);
  fp.bytes(m_data.atomic_numbers.data(),
           m_data.atomic_numbers.size() * sizeof(int));
  auto fit = [&fp](const std::vector<Table> &tables, int n, double h) {
    fp.f64(h);
    std::vector<Spline> out;
    for (const Table &t : tables) {
      fp.bytes(t.data(), t.size() * sizeof(double));
      out.push_back(spline(t, n, h));
    }
    return out;
  };
  m_frho = fit(m_data.frho, m_data.nrho, m_data.drho);
  m_rhor = fit(m_data.rhor, m_data.nr, m_data.dr);
  m_z2r = fit(m_data.z2r, m_data.nr, m_data.dr);
  m_paramsKey = fp.h;
}

void EAMTablePot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const std::size_t n = in.nAtoms;
  const std::size_t nel = m_data.atomic_numbers.size();
  const auto &zs = m_data.atomic_numbers;
  double *F = out->F;
  std::fill_n(F, 3 * n, 0.0);
  out->variance = 0.0;
  if (in.atmnrs == nullptr && nel > 1) {
    throw std::invalid_argument("an alloy EAM table needs atomic numbers");
  }
  std::vector<std::size_t> el(n, 0);
  for (std::size_t i = 0; in.atmnrs != nullptr && i < n; ++i) {
    const auto hit = std::find(zs.begin(), zs.end(), in.atmnrs[i]);
    if (hit == zs.end()) {
      throw std::invalid_argument("atomic number not in the EAM table");
    }
    el[i] = static_cast<std::size_t>(hit - zs.begin());
  }

  static constexpr double free_box[9] = {1.0e6, 0.0, 0.0, 0.0,  1.0e6,
                                         0.0,   0.0, 0.0, 1.0e6};
  const double *h = m_periodic ? in.box : free_box;
  if (h == nullptr) {
    throw std::invalid_argument("periodic EAM table needs a cell");
  }
  for (int k = 0; m_periodic && k < 3; ++k) {
    // Width along k is the cell volume over the area of the other two rows.
    const double *u = h + 3 * ((k + 1) % 3);
    const double *v = h + 3 * ((k + 2) % 3);
    const double w[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                         u[0] * v[1] - u[1] * v[0]};
    const double *a = h + 3 * k;
    const double vol = std::abs(a[0] * w[0] + a[1] * w[1] + a[2] * w[2]);
    if (!(vol > 2.0 * m_data.cut * std::hypot(w[0], w[1], w[2]))) {
      throw std::invalid_argument("EAM table cutoff exceeds half the cell");
    }
  }
  nlist::CachedPairList::Options opt;
  opt.cutoff = m_data.cut;
  opt.periodic = {m_periodic, m_periodic, m_periodic};

  const double cut2 = m_data.cut * m_data.cut;
  const double rdr = 1.0 / m_data.dr;
  // Interval and offset of r on the pair grid, as PairEAM::compute.
  auto knot = [rdr, nr = m_data.nr](double r, std::size_t &m) {
    const double p = r * rdr + 1.0;
    const int k = std::min(static_cast<int>(p), nr - 1);
    m = static_cast<std::size_t>(k);
    return std::min(p - k, 1.0);
  };
  auto &pool = nlist::PairListCache::global();
  std::vector<double> rho(n, 0.0);
  pool.evaluate(in.pos, n, h, opt,
                [&](int32_t i, int32_t j, double, double, double, double r2) {
                  if (!(r2 < cut2)) {
                    return;
                  }
                  std::size_t m = 0;
                  const double p = knot(std::sqrt(r2), m);
                  rho[i] += value(m_rhor[el[j] * nel + el[i]][m], p);
                  rho[j] += value(m_rhor[el[i] * nel + el[j]][m], p);
                });

  const double rhomax = (m_data.nrho - 1) * m_data.drho;
  const double rdrho = 1.0 / m_data.drho;
  std::vector<double> fp(n, 0.0);
  double energy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double p = rho[i] * rdrho + 1.0;
    const int k = std::max(1, std::min(static_cast<int>(p), m_data.nrho - 1));
    p = std::min(p - k, 1.0);
    const auto &c = m_frho[el[i]][static_cast<std::size_t>(k)];
    fp[i] = slope(c, p);
    energy += value(c, p) + fp[i] * std::max(rho[i] - rhomax, 0.0);
  }

  pool.evaluate(
      in.pos, n, h, opt,
      [&](int32_t i, int32_t j, double dx, double dy, double dz, double r2) {
        if (!(r2 < cut2)) {
          return;
        }
        const double r = std::sqrt(r2);
        std::size_t m = 0;
        const double p = knot(r, m);
        const std::size_t a = el[i];
        const std::size_t b = el[j];
        const double rhoip = slope(m_rhor[a * nel + b][m], p);
        const double rhojp = slope(m_rhor[b * nel + a][m], p);
        const auto &z = m_z2r[a * nel + b][m];
        const double recip = 1.0 / r;
        const double phi = value(z, p) * recip;
        const double phip = slope(z, p) * recip - phi * recip;
        const double fpair = -(fp[i] * rhojp + fp[j] * rhoip + phip) * recip;
        energy += phi;
        F[3 * i] += dx * fpair;
        F[3 * i + 1] += dy * fpair;
        F[3 * i + 2] += dz * fpair;
        F[3 * j] -= dx * fpair;
        F[3 * j + 1] -= dy * fpair;
        F[3 * j + 2] -= dz * fpair;
      });
  out->energy = energy;
}

} // namespace rgpot
