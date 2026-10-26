// MIT License
// Copyright 2023--present rgpot developers
//
// Local relaxation of a deposited Sutton-Chen geometry after a uniform
// rescaling. Prints the initial energy, the relaxed energy, and the
// largest force component.

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "rgpot/SuttonChen/SuttonChenScales.hpp"

namespace {

struct Geometry {
  std::vector<double> x;
};

Geometry read_points(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  std::string line;
  std::getline(in, line);
  Geometry out;
  while (std::getline(in, line)) {
    std::istringstream row(line);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (row >> x >> y >> z) {
      out.x.push_back(x);
      out.x.push_back(y);
      out.x.push_back(z);
    }
  }
  return out;
}

double max_abs(const std::vector<double> &v) {
  double m = 0.0;
  for (double value : v) {
    m = std::max(m, std::abs(value));
  }
  return m;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: relax_sutton_chen pt|au scale geometry\n";
    return 2;
  }
  const std::string which = argv[1];
  const double scale = std::stod(argv[2]);
  auto cfg = which == "au" ? rgpot::sutton_chen_gold()
                           : rgpot::sutton_chen_platinum();
  if (which != "au" && which != "pt") {
    std::cerr << "metal must be pt or au\n";
    return 2;
  }
  cfg.cutoff = 0.0;
  cfg.periodic = false;
  auto points = read_points(argv[3]);
  for (double &value : points.x) {
    value *= scale;
  }
  const auto natoms = static_cast<int32_t>(points.x.size() / 3);
  std::vector<double> forces(points.x.size(), 0.0);
  double energy = 0.0;
  const rgpot_sc_config raw{cfg.epsilon, cfg.c, cfg.a, cfg.n, cfg.m, cfg.cutoff,
                            0};
  if (rgpot_sutton_chen_energy_forces(&raw, natoms, points.x.data(), nullptr, &energy,
                              forces.data()) != 0) {
    return 1;
  }
  const double initial = energy;
  double step = 1.0e-3 * cfg.a;
  for (int iter = 0; iter < 4000; ++iter) {
    if (max_abs(forces) < 1.0e-8) {
      break;
    }
    std::vector<double> trial = points.x;
    for (std::size_t i = 0; i < trial.size(); ++i) {
      trial[i] += step * forces[i];
    }
    double next = 0.0;
    std::vector<double> next_forces(trial.size(), 0.0);
    if (rgpot_sutton_chen_energy_forces(&raw, natoms, trial.data(), nullptr, &next,
                                next_forces.data()) != 0) {
      return 1;
    }
    if (next < energy) {
      points.x.swap(trial);
      forces.swap(next_forces);
      energy = next;
      step *= 1.2;
    } else {
      step *= 0.5;
    }
  }
  std::cout.setf(std::ios::fixed);
  std::cout.precision(10);
  std::cout << "metal: " << which << "\n";
  std::cout << "scale: " << scale << "\n";
  std::cout << "epsilon: " << cfg.epsilon << "\n";
  std::cout << "a: " << cfg.a << "\n";
  std::cout << "atoms: " << natoms << "\n";
  std::cout << "initial_energy: " << initial << "\n";
  std::cout << "relaxed_energy: " << energy << "\n";
  std::cout << "max_force: " << max_abs(forces) << "\n";
  return 0;
}
