// MIT License
// Copyright 2023--present rgpot developers
//
// Multi-rank host for NWChemPot: proves host-owned MPI through the rgpot
// frontend (same public nwchemc_energy_gradient path as nwchemc's timer).
//
// Contract: every MPI rank must call force with the same geometry/params.
// Cap'n Proto carries method/geometry only — not MPI ranks or communicators.
//
// Usage:
//   mpirun -np P nwchem_mpi_force_host [options]
//
// Options:
//   --system h2|water|benzene   built-in geometry (default: h2)
//   --geom PATH                 file: n_atoms then "Z x y z" (Angstrom) lines
//   --basis NAME                default sto-3g
//   --theory NAME               default scf
//   --scf-type NAME             default rhf
//   --engine PATH               libnwchemc.so (else NWCHEMC_LIBRARY / …)
//
// Rank 0 prints one machine-readable line:
//   rgpot_mpi_force system=… ranks=P wall_s=… energy_ev=… maxabs_f=… ok=…
//
// Environment for the engine (same as potserv / eOn):
//   NWCHEMC_LIBRARY, RGPOT_NWCHEMC_ENGINE, NWCHEM_TOP, LD_LIBRARY_PATH, …

#include <mpi.h>

#include <capnp/message.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "rgpot/NWChemPot/NWChemPot.hpp"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;

namespace {

struct Geom {
  std::string label;
  std::vector<int> Z;
  std::vector<double> xyz; // n*3 Angstrom
};

static Geom geom_h2() {
  Geom g;
  g.label = "h2";
  g.Z = {1, 1};
  g.xyz = {0.0, 0.0, -0.3707, 0.0, 0.0, 0.3707};
  return g;
}

static Geom geom_water() {
  Geom g;
  g.label = "water";
  g.Z = {8, 1, 1};
  g.xyz = {0.0, 0.0, 0.1173, 0.0, 0.7572, -0.4692, 0.0, -0.7572, -0.4692};
  return g;
}

static Geom geom_benzene() {
  Geom g;
  g.label = "benzene";
  g.Z.assign(12, 0);
  g.xyz.assign(36, 0.0);
  const double cc = 1.397;
  const double ch = 1.084;
  constexpr double pi = 3.14159265358979323846;
  for (int i = 0; i < 6; ++i) {
    const double a = (pi / 3.0) * static_cast<double>(i);
    g.Z[static_cast<size_t>(i)] = 6;
    g.xyz[static_cast<size_t>(3 * i)] = cc * std::cos(a);
    g.xyz[static_cast<size_t>(3 * i + 1)] = cc * std::sin(a);
    g.xyz[static_cast<size_t>(3 * i + 2)] = 0.0;
    g.Z[static_cast<size_t>(6 + i)] = 1;
    g.xyz[static_cast<size_t>(3 * (6 + i))] = (cc + ch) * std::cos(a);
    g.xyz[static_cast<size_t>(3 * (6 + i) + 1)] = (cc + ch) * std::sin(a);
    g.xyz[static_cast<size_t>(3 * (6 + i) + 2)] = 0.0;
  }
  return g;
}

static bool load_geom_file(const std::string &path, Geom *out) {
  std::ifstream in(path);
  if (!in)
    return false;
  int n = 0;
  if (!(in >> n) || n < 1 || n > 512)
    return false;
  out->label = path;
  out->Z.assign(static_cast<size_t>(n), 0);
  out->xyz.assign(static_cast<size_t>(n) * 3u, 0.0);
  for (int i = 0; i < n; ++i) {
    int z = 0;
    double x = 0, y = 0, zz = 0;
    if (!(in >> z >> x >> y >> zz))
      return false;
    out->Z[static_cast<size_t>(i)] = z;
    out->xyz[static_cast<size_t>(3 * i)] = x;
    out->xyz[static_cast<size_t>(3 * i + 1)] = y;
    out->xyz[static_cast<size_t>(3 * i + 2)] = zz;
  }
  return true;
}

static void usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: mpirun -np P %s [--system h2|water|benzene] "
               "[--geom file] [--basis B] [--theory T] [--scf-type S] "
               "[--engine PATH]\n",
               argv0);
}

static const char *env_or(const char *key, const char *fallback) {
  const char *v = std::getenv(key);
  return (v && v[0]) ? v : fallback;
}

} // namespace

int main(int argc, char **argv) {
  int rank = 0, nprocs = 1;
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Init failed\n");
    return 1;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  std::string system_name = "h2";
  std::string geom_path;
  std::string basis = env_or("RGPOT_NWCHEM_BASIS", "sto-3g");
  std::string theory = env_or("RGPOT_NWCHEM_THEORY", "scf");
  std::string scf_type = env_or("RGPOT_NWCHEM_SCF_TYPE", "rhf");
  std::string engine_path;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char * /*flag*/) -> std::string {
      if (i + 1 >= argc) {
        if (rank == 0)
          usage(argv[0]);
        MPI_Finalize();
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--system")
      system_name = need("--system");
    else if (a == "--geom")
      geom_path = need("--geom");
    else if (a == "--basis")
      basis = need("--basis");
    else if (a == "--theory")
      theory = need("--theory");
    else if (a == "--scf-type")
      scf_type = need("--scf-type");
    else if (a == "--engine")
      engine_path = need("--engine");
    else if (a == "-h" || a == "--help") {
      if (rank == 0)
        usage(argv[0]);
      MPI_Finalize();
      return 0;
    } else {
      if (rank == 0) {
        std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
        usage(argv[0]);
      }
      MPI_Finalize();
      return 2;
    }
  }

  Geom geom;
  if (!geom_path.empty()) {
    if (!load_geom_file(geom_path, &geom)) {
      if (rank == 0)
        std::fprintf(stderr, "failed to load geom %s\n", geom_path.c_str());
      MPI_Finalize();
      return 1;
    }
  } else if (system_name == "water") {
    geom = geom_water();
  } else if (system_name == "benzene") {
    geom = geom_benzene();
  } else {
    geom = geom_h2();
    system_name = "h2";
  }

  // Identical inputs on every rank (SPMD PEF contract).
  // Scope pot so ~NWChemPot / engine GA+MPI teardown runs *before* host
  // MPI_Finalize (embed may call mpi_finalize when it owns the world).
  int ok_min = 0;
  {
    ::capnp::MallocMessageBuilder msg;
    auto p = msg.initRoot<::NWChemParams>();
    p.setBasis(basis);
    p.setTheory(theory);
    p.setScfType(scf_type);
    p.setCharge(0);
    p.setMultiplicity(1);
    p.setTask("gradient");
    if (!engine_path.empty())
      p.setEnginePath(engine_path);

    rgpot::NWChemPot pot(p.asReader());
    if (!pot.available()) {
      if (rank == 0)
        std::fprintf(stderr,
                     "NWChem engine not loaded (set --engine or "
                     "NWCHEMC_LIBRARY / RGPOT_NWCHEMC_ENGINE)\n");
      MPI_Finalize();
      return 1;
    }

    const int n = static_cast<int>(geom.Z.size());
    AtomMatrix positions(n, 3);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < 3; ++j)
        positions(i, j) = geom.xyz[static_cast<size_t>(3 * i + j)];
    std::vector<int> atmtypes = geom.Z;
    // Large vacuum box (Å); molecular PEF, not periodic SCF.
    std::array<std::array<double, 3>, 3> box = {
        {{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();
    int ok = 0;
    double energy = 0.0;
    double maxabs_f = 0.0;
    std::string err;
    try {
      auto [e, forces] = pot(positions, atmtypes, box);
      energy = e;
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < 3; ++j) {
          const double a = std::fabs(forces(i, j));
          if (a > maxabs_f)
            maxabs_f = a;
        }
      ok = (std::isfinite(energy) && std::isfinite(maxabs_f)) ? 1 : 0;
    } catch (const std::exception &ex) {
      err = ex.what();
      ok = 0;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_s =
        std::chrono::duration<double>(t1 - t0).count();

    // All ranks must succeed for a valid PEF (GA collectives).
    ok_min = ok;
    MPI_Allreduce(MPI_IN_PLACE, &ok_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
      if (!err.empty())
        std::fprintf(stderr, "error: %s\n", err.c_str());
      std::printf("rgpot_mpi_force system=%s n_atoms=%d ranks=%d wall_s=%.6f "
                  "energy_ev=%.12g maxabs_f=%.6e ok=%d basis=%s theory=%s\n",
                  geom.label.c_str(), n, nprocs, wall_s, energy, maxabs_f,
                  ok_min, basis.c_str(), theory.c_str());
      std::fflush(stdout);
    }
  }

  // Host-owned world: only finalize if still initialized (embed may have).
  {
    int flag = 0;
    MPI_Finalized(&flag);
    if (!flag)
      MPI_Finalize();
  }
  return ok_min ? 0 : 1;
}
