// MIT License
#include "tools/MpiHost.hpp"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::atomic<bool> reject_payload = false;
int world_rank = 0;
int world_size = 1;
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void inspect_finalized_runtime() {
  int finalized = 0;
  MPI_Finalized(&finalized);
  std::printf("observer rank=%d finalized=%d\n", world_rank, finalized);
  if (!finalized)
    std::abort();
}
void engine_exit_handler() {
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (finalized) {
    std::fprintf(stderr, "runtime finalized before engine exit handler\n");
    std::abort();
  }
  int one = 1;
  int total = 0;
  MPI_Allreduce(&one, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  std::printf("engine exit rank=%d participants=%d\n", world_rank, total);
  if (total != world_size)
    std::abort();
}
}

void *operator new(std::size_t bytes) {
  if (bytes == 4096 * sizeof(double) && reject_payload.exchange(false))
    throw std::bad_alloc();
  if (void *memory = std::malloc(bytes ? bytes : 1))
    return memory;
  throw std::bad_alloc();
}
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }

int main(int argc, char **argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  try {
    if (mode == "owned") {
      require(std::atexit(inspect_finalized_runtime) == 0, "cannot register observer");
      {
        rgpot::tools::MpiHost host(argc, argv);
        world_rank = host.rank();
        world_size = host.size();
        require(std::atexit(engine_exit_handler) == 0, "cannot register engine handler");
      }
      int finalized = 0;
      MPI_Finalized(&finalized);
      require(!finalized, "host ended MPI before engine exit handlers");
      return 0;
    }
    require(mode == "borrowed", "unknown contract mode");
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    require(world_size == 4, "borrowed contract needs four ranks");
    const int color = world_rank / 2;
    MPI_Comm borrowed = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &borrowed);
    {
      rgpot::tools::MpiHost host(argc, argv, borrowed);
      require(host.size() == 2, "borrowed communicator membership changed");
      bool received_failure = false;
      try {
        host.collective([&] {
          if (world_rank == 1)
            throw std::runtime_error("");
        });
      } catch (const std::runtime_error &error) {
        require(std::string(error.what()).find("rank 1") != std::string::npos,
                "worker exception did not identify its rank");
        received_failure = true;
      }
      require(received_failure == (color == 0),
              "failure crossed the borrowed communicator boundary");
      std::vector<double> payload;
      if (host.rank() == 0)
        payload.assign(4096, 17.0 + color);
      reject_payload = world_rank == 1;
      received_failure = false;
      try {
        host.broadcast(payload);
      } catch (const std::runtime_error &error) {
        require(std::string(error.what()).find("rank 1") != std::string::npos,
                "allocation error did not reach the source rank");
        received_failure = true;
      }
      require(!reject_payload, "receive allocation failure was not exercised");
      require(received_failure == (color == 0),
              "allocation failure did not agree within the communicator");
      host.broadcast(payload);
      require(payload.size() == 4096, "transfer failed after allocation recovery");
      for (double value : payload)
        require(value == 17.0 + color, "payload crossed communicator groups");
    }
    int finalized = 0;
    MPI_Finalized(&finalized);
    require(!finalized, "host finalized a borrowed runtime");
    int retained_size = 0;
    MPI_Comm_size(borrowed, &retained_size);
    require(retained_size == 2, "host freed its borrowed communicator");
    MPI_Comm_free(&borrowed);
    MPI_Finalize();
    std::printf("borrowed contract rank=%d passed\n", world_rank);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    int initialized = 0;
    MPI_Initialized(&initialized);
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (initialized && !finalized)
      MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
