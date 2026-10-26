// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/CPMDPot/CPMDPot.hpp"
#include "rgpot/CalculatorGroup.hpp"

#include <mpi.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <unistd.h>

namespace {

void guestChecker() {
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (finalized)
    _exit(3);
  MPI_Finalize();
}

void redirectRankLog(int rank) {
  const char *dir = std::getenv("RGPOT_RANK_LOG_DIR");
  if (!dir || dir[0] == '\0')
    return;
  char path[4096];
  const int n = std::snprintf(path, sizeof path, "%s/rank-%d.log", dir, rank);
  if (n <= 0 || static_cast<std::size_t>(n) >= sizeof path)
    MPI_Abort(MPI_COMM_WORLD, 2);
  const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0)
    MPI_Abort(MPI_COMM_WORLD, 2);
  if (dup2(fd, STDOUT_FILENO) < 0 || dup2(fd, STDERR_FILENO) < 0)
    MPI_Abort(MPI_COMM_WORLD, 2);
  if (fd > STDERR_FILENO)
    close(fd);
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);
}

// Both ranks form one calculator; each fails its force call, and the
// calculator's error exchange prints rank 0's message on rank 1 too.
int runAbort() {
  const rgpot::CalculatorGroup group = rgpot::bindCalculators(2);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  redirectRankLog(rank);
  if (group.ranks != 2 || group.index != 0) {
    std::fprintf(stderr, "bind failed index=%d ranks=%d\n", group.index,
                 group.ranks);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  rgpot::CPMDPot pot;
  if (!pot.available()) {
    std::fprintf(stderr, "engine not loaded\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  double pos[3] = {0.0, 0.0, 0.0};
  int atm = 8;
  double box[9] = {20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
  double forces[3] = {0.0, 0.0, 0.0};
  rgpot::ForceOut out{};
  out.F = forces;
  try {
    if (rank == 0) {
      const rgpot::ForceInput in{
          .nAtoms = 0, .pos = pos, .atmnrs = &atm, .box = box};
      pot.forceImpl(in, &out);
    } else {
      const rgpot::ForceInput in{
          .nAtoms = 1, .pos = nullptr, .atmnrs = &atm, .box = box};
      pot.forceImpl(in, &out);
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "exception after abort: %s\n", ex.what());
    return 4;
  }
  std::fprintf(stderr, "forceImpl returned\n");
  return 4;
}

int runShareBad() {
  const rgpot::CalculatorGroup group = rgpot::bindCalculators(1);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  redirectRankLog(rank);
  if (group.index < 0 || group.ranks != 1) {
    std::fprintf(stderr, "bind failed index=%d ranks=%d\n", group.index,
                 group.ranks);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  std::array<double, 4> buf{1.0, -2.0, 3.5, 0.25};
  const int owner = rank == 0 ? 0 : 5;
  const int shared =
      rgpot::shareFromCalculator(owner, buf.data(), sizeof(buf));
  std::fprintf(stderr, "share returned %d\n", shared);
  return 4;
}

int runShareOk() {
  const rgpot::CalculatorGroup group = rgpot::bindCalculators(1);
  // mpirun treats an initialized process that skips MPI_Finalize as a failed job.
  rgpot::finalizeMpiAtExit();
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  redirectRankLog(rank);
  if (group.index < 0 || group.ranks != 1) {
    std::fprintf(stderr, "bind failed index=%d ranks=%d\n", group.index,
                 group.ranks);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  std::array<double, 4> buf{0.0, 0.0, 0.0, 0.0};
  if (rank == 0)
    buf = {1.0, -2.0, 3.5, 0.25};
  const int shared = rgpot::shareFromCalculator(0, buf.data(), sizeof(buf));
  if (shared != 1 || buf != std::array<double, 4>{1.0, -2.0, 3.5, 0.25}) {
    std::fprintf(stderr, "share-ok failed shared=%d\n", shared);
    return 4;
  }
  if (rank == 1) {
    std::fprintf(stderr, "share-ok %.6f %.6f %.6f %.6f\n", buf[0], buf[1],
                 buf[2], buf[3]);
  }
  return 0;
}

// Four ranks as two calculators of two. Only calculator 0 gets a system
// (an uneven batch), then every rank takes calculator 0's result from
// shareFromCalculator. The force call's collectives stay inside
// calculator 0, so calculator 1 waits in the share and nothing hangs.
int runUneven() {
  const rgpot::CalculatorGroup group = rgpot::bindCalculators(2);
  rgpot::finalizeMpiAtExit();
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  redirectRankLog(rank);
  if (group.ranks != 2 || group.index < 0 || group.world_size != 4) {
    std::fprintf(stderr, "bind failed index=%d ranks=%d\n", group.index,
                 group.ranks);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  rgpot::CPMDPot pot;
  if (!pot.available()) {
    std::fprintf(stderr, "engine not loaded\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  std::array<double, 4> result{0.0, 0.0, 0.0, 0.0};
  if (group.index == 0) {
    double pos[3] = {0.0, 0.0, 0.0};
    int atm = 8;
    double box[9] = {20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
    rgpot::ForceOut out{};
    out.F = result.data() + 1;
    const rgpot::ForceInput in{
        .nAtoms = 1, .pos = pos, .atmnrs = &atm, .box = box};
    pot.forceImpl(in, &out);
    result[0] = out.energy;
  }
  const std::array<double, 4> mine = result;
  if (rgpot::shareFromCalculator(0, result.data(), sizeof(result)) != 1) {
    std::fprintf(stderr, "uneven share failed\n");
    return 4;
  }
  if (group.index == 0 && result != mine) {
    std::fprintf(stderr, "uneven share changed the owner's result\n");
    return 4;
  }
  std::fprintf(stderr, "uneven ok %.6f\n", result[0]);
  return 0;
}

int runOwner() {
  rgpot::bindCalculators(1);
  rgpot::finalizeMpiAtExit();
  return 0;
}

int runGuest() {
  MPI_Init(nullptr, nullptr);
  std::atexit(guestChecker);
  rgpot::bindCalculators(1);
  rgpot::finalizeMpiAtExit();
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "abort";
  if (std::strcmp(mode, "finalize-owner") == 0)
    return runOwner();
  if (std::strcmp(mode, "finalize-guest") == 0)
    return runGuest();
  if (std::strcmp(mode, "share-bad") == 0)
    return runShareBad();
  if (std::strcmp(mode, "share-ok") == 0)
    return runShareOk();
  if (std::strcmp(mode, "uneven") == 0)
    return runUneven();
  return runAbort();
}
