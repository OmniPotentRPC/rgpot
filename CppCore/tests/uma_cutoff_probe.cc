// Copyright (c) rgpot developers
//
// Claisen pair past the molecular trace cutoff.
//
// The attached endpoints do not contain the dimer frame that died.
// Frame 0's nearest contact (C4-H12) is 0.5576 file units. Times
// BOHR_TO_ANGSTROM that is 0.295 angstrom. Neither reading is a bond,
// and the longest pair in the frame is under 4, so no uniform scale
// puts atoms 5 and 11 at 10.197616276524268 angstrom without also
// pushing other pairs past 10 angstrom. The geometry here is frame 0
// with those two atoms moved equally along the line that joins them
// until the distance is the logged value. The raw numbers are kept as
// angstrom, the unit of the harness distance and of the engine cutoff.
//
// The package is scripts/export_cutoff_probe.py. It is traced at
// 14*13 edges, records cutoff 10 (the old 0.4*25 trace cutoff) and
// model_cutoff 6 (the envelope). A force call must succeed, including
// while the pair steps across 10 angstrom, and the energy must match
// the same sum with every pair past model_cutoff removed.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "rgpot/UmaPot/UmaPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

namespace {

constexpr int kAtoms = 14;
constexpr double kTarget = 10.197616276524268;
constexpr double kModelCutoff = 6.0;
constexpr double kGraphCutoff = 10.0;
constexpr double kBox = 25.0;

// baker/17_claisen endpoint frame 0, file units.
constexpr double kFrame0[kAtoms][3] = {
    {0.96601080, -0.33392482, 0.69680887},
    {0.27933941, -0.39005711, 0.64163902},
    {-0.08738951, -0.63637662, -0.00872396},
    {-0.37571458, 0.43525148, -0.51394545},
    {-0.80817203, 0.99043931, -0.67109528},
    {-0.63111841, -0.17400821, -0.24752542},
    {1.32236146, -0.48145900, 0.27432843},
    {1.21485645, -0.15499709, 1.17824275},
    {-0.06006874, -0.24234469, 1.07435274},
    {-0.36090676, -1.14831179, 0.10468814},
    {0.29610618, -0.73196832, -0.42999683},
    {-0.58501592, 1.45189783, -0.90637601},
    {-1.35391476, 0.96676073, -0.55911516},
    {0.18362638, 0.44909833, -0.63328183},
};

const std::vector<int> kZ = {6, 6, 6, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1};

struct Eval {
  double energy = 0.0;
  std::vector<double> force;
};

int fail(const std::string &message) {
  std::cerr << "FAIL " << message << "\n";
  return 1;
}

double dist3(const double *a, const double *b) {
  const double dx = a[0] - b[0];
  const double dy = a[1] - b[1];
  const double dz = a[2] - b[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Atoms 5 and 11 sit `separation` angstrom apart along their frame-0
// direction, midpoint held fixed. Every other atom stays on the endpoint.
std::vector<double> geometry(double separation) {
  std::vector<double> pos(3 * kAtoms);
  for (int i = 0; i < kAtoms; ++i)
    for (int d = 0; d < 3; ++d)
      pos[3 * i + d] = kFrame0[i][d];
  double mid[3], step[3];
  double length2 = 0.0;
  for (int d = 0; d < 3; ++d) {
    mid[d] = 0.5 * (pos[3 * 5 + d] + pos[3 * 11 + d]);
    step[d] = pos[3 * 11 + d] - pos[3 * 5 + d];
    length2 += step[d] * step[d];
  }
  const double length = std::sqrt(length2);
  const double half = 0.5 * separation;
  for (int d = 0; d < 3; ++d) {
    const double u = step[d] / length;
    pos[3 * 5 + d] = mid[d] - u * half;
    pos[3 * 11 + d] = mid[d] + u * half;
  }
  return pos;
}

int directed_below(const std::vector<double> &pos, double cutoff) {
  int count = 0;
  for (int i = 0; i < kAtoms; ++i) {
    for (int j = 0; j < kAtoms; ++j) {
      if (i == j)
        continue;
      if (dist3(&pos[3 * i], &pos[3 * j]) < cutoff)
        ++count;
    }
  }
  return count;
}

// Envelope shared with scripts/export_cutoff_probe.py. `drop_beyond`
// skips pairs at or past the model cutoff. Those terms are zero, so the
// two sums match.
Eval reference(const std::vector<double> &pos, bool drop_beyond) {
  Eval out;
  out.force.assign(3 * kAtoms, 0.0);
  const double c2 = kModelCutoff * kModelCutoff;
  for (int center = 0; center < kAtoms; ++center) {
    for (int neighbor = 0; neighbor < kAtoms; ++neighbor) {
      if (center == neighbor)
        continue;
      double d[3];
      double r2 = 0.0;
      for (int a = 0; a < 3; ++a) {
        d[a] = pos[3 * neighbor + a] - pos[3 * center + a];
        r2 += d[a] * d[a];
      }
      if (!(r2 < c2)) {
        // The edge is in the list and the envelope is zero. Skipping it
        // is the same sum.
        if (!drop_beyond)
          out.energy += 0.0;
        continue;
      }
      const double x = r2 / c2;
      out.energy += (1.0 - x) * (1.0 - x);
      const double scale = (-4.0 / c2) * (1.0 - x);
      for (int a = 0; a < 3; ++a) {
        const double g = scale * d[a];
        out.force[3 * neighbor + a] -= g;
        out.force[3 * center + a] += g;
      }
    }
  }
  return out;
}

Eval evaluate(rgpot::UmaPot &pot, const std::vector<double> &pos) {
  rgpot::types::AtomMatrix positions(kAtoms, 3);
  for (int i = 0; i < kAtoms; ++i)
    for (int d = 0; d < 3; ++d)
      positions(static_cast<size_t>(i), static_cast<size_t>(d)) =
          pos[3 * i + d];
  const std::array<std::array<double, 3>, 3> box{{
      {kBox, 0.0, 0.0},
      {0.0, kBox, 0.0},
      {0.0, 0.0, kBox},
  }};
  auto [energy, forces, variance] = pot(positions, kZ, box);
  (void)variance;
  Eval out;
  out.energy = energy;
  out.force.resize(3 * kAtoms);
  for (int i = 0; i < kAtoms; ++i)
    for (int d = 0; d < 3; ++d)
      out.force[3 * i + d] =
          forces(static_cast<size_t>(i), static_cast<size_t>(d));
  return out;
}

bool close_eval(const Eval &got, const Eval &want, double e_tol, double f_tol,
                const std::string &label) {
  const double de = std::abs(got.energy - want.energy);
  double df = 0.0;
  for (size_t i = 0; i < got.force.size(); ++i)
    df = std::max(df, std::abs(got.force[i] - want.force[i]));
  std::cout << label << " dE=" << de << " max|dF|=" << df << "\n";
  if (de > e_tol || df > f_tol) {
    std::cerr << "FAIL " << label << " dE=" << de << " max|dF|=" << df << "\n";
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const std::string pt2 = [&] {
    if (const char *env = std::getenv("RGPOT_UMA_CUTOFF_PT2"))
      if (*env)
        return std::string(env);
    const char *out = argc > 1 ? argv[1] : "/tmp/rgpot-uma-cutoff-probe.pt2";
    const std::string cmd =
        "python3 scripts/export_cutoff_probe.py --out '" + std::string(out) +
        "'";
    if (std::system(cmd.c_str()) != 0)
      return std::string();
    return std::string(out);
  }();
  if (pt2.empty())
    return fail("could not build the cutoff probe package");

  const double nn = [&] {
    double best = 1e300;
    for (int i = 0; i < kAtoms; ++i)
      for (int j = i + 1; j < kAtoms; ++j)
        best = std::min(best, dist3(kFrame0[i], kFrame0[j]));
    return best;
  }();
  std::cout << "endpoint_nn_file=" << nn
            << " as_angstrom=" << nn
            << " as_bohr_to_A=" << nn * rgpot::units::BOHR_TO_ANGSTROM << "\n";
  if (!(nn * rgpot::units::BOHR_TO_ANGSTROM < 0.4) || !(nn < 0.8))
    return fail("endpoint unit check");

  const auto base = geometry(kTarget);
  const double pair = dist3(&base[3 * 5], &base[3 * 11]);
  const int below = directed_below(base, kGraphCutoff);
  std::cout << "pair_5_11_A=" << pair << " directed_below_10=" << below
            << " compiled_edges=" << kAtoms * (kAtoms - 1) << "\n";
  if (std::abs(pair - kTarget) > 1e-12)
    return fail("pair 5-11 is not the logged distance");
  if (below != kAtoms * (kAtoms - 1) - 2)
    return fail("10 angstrom list does not drop exactly one unordered pair");

  const Eval kept = reference(base, false);
  const Eval dropped = reference(base, true);
  if (!close_eval(kept, dropped, 1e-12, 1e-12, "remove_pair_past_model_cutoff"))
    return 1;

  rgpot::UmaConfig cfg;
  cfg.model_path = pt2;
  cfg.device = "cpu";
  cfg.task_name = "omol";
  cfg.charge = 0;
  cfg.spin = 1;
  try {
    rgpot::UmaPot pot(cfg);
    // One calculator. Steps at and under 10 angstrom still have 182
    // edges. The logged distance is the first step past that cutoff:
    // the parent list drops both directions and the sequence stops.
    const double steps[] = {8.0, 9.5, 9.9, 10.0, kTarget, 10.5, 11.0};
    for (double separation : steps) {
      const auto pos = geometry(separation);
      Eval step;
      try {
        step = evaluate(pot, pos);
      } catch (const std::exception &e) {
        return fail("step separation " + std::to_string(separation) + ": " +
                    e.what());
      }
      if (!close_eval(step, reference(pos, true), 1e-12, 1e-12,
                      "step_" + std::to_string(separation)))
        return 1;
    }

    const double fd_at[] = {9.5, 10.0, kTarget};
    constexpr double kH = 1e-5;
    constexpr int kAtom = 11;
    constexpr int kCoord = 0;
    for (double separation : fd_at) {
      auto pos = geometry(separation);
      Eval center;
      try {
        center = evaluate(pot, pos);
      } catch (const std::exception &e) {
        return fail("fd center " + std::to_string(separation) + ": " + e.what());
      }
      pos[3 * kAtom + kCoord] += kH;
      const double ep = evaluate(pot, pos).energy;
      pos[3 * kAtom + kCoord] -= 2.0 * kH;
      const double em = evaluate(pot, pos).energy;
      const double fd = (ep - em) / (2.0 * kH);
      const double force = center.force[3 * kAtom + kCoord];
      const double err = std::abs(fd + force);
      std::cout << "fd separation=" << separation << " |dE/dx+F|=" << err
                << "\n";
      if (err > 1e-8)
        return fail("finite difference at " + std::to_string(separation));
    }
  } catch (const std::exception &e) {
    return fail(std::string("load: ") + e.what());
  }
  std::cout << "PASS uma cutoff probe\n";
  return 0;
}
