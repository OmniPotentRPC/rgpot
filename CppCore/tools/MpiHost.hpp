// MIT License
// Copyright 2023--present rgpot developers
#pragma once

#include <mpi.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace rgpot::tools {

// MPI belongs to these explicit host executables. The core library and serial
// server retain their separate lazy-loading contract.
class MpiHost {
public:
  MpiHost(int &argc, char **&argv, MPI_Comm communicator = MPI_COMM_WORLD)
      : communicator_(communicator) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized)
      throw std::runtime_error("MPI host cannot use a finalized runtime");
    if (!initialized) {
      if (communicator != MPI_COMM_WORLD)
        throw std::runtime_error("MPI host requires a live borrowed communicator");
      if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Init failed");
      // Engine exit handlers register after this handler, so their collectives
      // finish before the host-owned runtime is finalized. A borrowed runtime
      // has no host exit handler, and a borrowed communicator is never freed.
      if (std::atexit(finalizeOwnedRuntime) != 0)
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    check(MPI_Comm_rank(communicator_, &rank_));
    check(MPI_Comm_size(communicator_, &size_));
  }

  MpiHost(const MpiHost &) = delete;
  MpiHost &operator=(const MpiHost &) = delete;

  [[nodiscard]] int rank() const { return rank_; }
  [[nodiscard]] int size() const { return size_; }
  [[nodiscard]] MPI_Comm communicator() const { return communicator_; }

  // Each rank calls the function once. It must return or throw on every rank;
  // collective operations inside an engine remain the engine's responsibility.
  template <class Function> void collective(Function &&function) const {
    std::array<char, 1024> error{};
    int failed = size_;
    try {
      std::forward<Function>(function)();
    } catch (const std::exception &exception) {
      failed = rank_;
      std::snprintf(error.data(), error.size(), "%s", exception.what());
    } catch (...) {
      failed = rank_;
      std::snprintf(error.data(), error.size(), "unknown host exception");
    }
    int first = size_;
    check(MPI_Allreduce(&failed, &first, 1, MPI_INT, MPI_MIN, communicator_));
    if (first == size_)
      return;
    check(MPI_Bcast(error.data(), static_cast<int>(error.size()), MPI_CHAR,
                    first, communicator_));
    std::array<char, 1088> message{};
    std::snprintf(message.data(), message.size(), "MPI host rank %d: %s",
                  first, error.data());
    throw std::runtime_error(message.data());
  }

  template <class Value> void broadcast(std::vector<Value> &values) const {
    static_assert(std::is_trivially_copyable_v<Value>);
    std::uint64_t count = values.size();
    check(MPI_Bcast(&count, 1, MPI_UINT64_T, 0, communicator_));
    // Agree on allocation failure before any rank enters the payload transfer.
    collective([&] {
      if (count > values.max_size() ||
          count > std::numeric_limits<std::size_t>::max() / sizeof(Value))
        throw std::length_error("MPI host payload exceeds addressable storage");
      values.resize(static_cast<std::size_t>(count));
    });
    std::size_t left = values.size() * sizeof(Value);
    auto *data = reinterpret_cast<unsigned char *>(values.data());
    while (left) {
      const int chunk = static_cast<int>(
          std::min(left, static_cast<std::size_t>(INT_MAX)));
      check(MPI_Bcast(data, chunk, MPI_BYTE, 0, communicator_));
      data += chunk;
      left -= chunk;
    }
  }

  void broadcastCommand(int &command) const {
    check(MPI_Bcast(&command, 1, MPI_INT, 0, communicator_));
  }

  [[nodiscard]] double maximum(double local) const {
    double result = 0;
    check(MPI_Allreduce(&local, &result, 1, MPI_DOUBLE, MPI_MAX, communicator_));
    return result;
  }

private:
  static void finalizeOwnedRuntime() {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized)
      MPI_Finalize();
  }

  void check(int status) const {
    if (status != MPI_SUCCESS)
      MPI_Abort(communicator_, status);
  }

  MPI_Comm communicator_;
  int rank_ = 0;
  int size_ = 1;
};

} // namespace rgpot::tools
