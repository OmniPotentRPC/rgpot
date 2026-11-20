// MIT License
// Copyright 2023--present rgpot developers
#include <atomic>
#include <catch2/catch_all.hpp>
#include <random>
#include <thread>
#include <vector>

// Include the C-API header
#include "rgpot/rpc/pot_bridge.h"

// Helper to connect to a running server process
const std::string HOST = "127.0.0.1";
const int PORT = 12345;

static bool require_potserv() {
  const char *v = std::getenv("RGPOT_REQUIRE_POTSERV");
  return v != nullptr && v[0] != '\0' && std::string(v) != "0" &&
         std::string(v) != "false";
}

// Helper to generate random double data
std::vector<double> gen_random_data(size_t size) {
  static std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  std::vector<double> data(size);
  for (auto &v : data)
    v = dist(rng);
  return data;
}

TEST_CASE("Bridge Lifecycle and Robustness", "[bridge][core]") {

  SECTION("Handle NULL gracefully") {
    // Passing actual NULL to init should return NULL
    PotClient *client = pot_client_init(nullptr, 9999);
    CHECK(client == nullptr);

    // Passing NULL to other functions should not crash
    pot_client_free(nullptr);

    double eng = 0;
    double f[3] = {0};
    int32_t atmnrs[1] = {1};
    double pos[3] = {0, 0, 0};
    double box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    CHECK(pot_calculate(nullptr, 1, pos, atmnrs, box, &eng, f) != 0);
  }

  SECTION("Lazy Connection Failure") {
    // A syntactically valid host returns a handle (Lazy Connection)
    PotClient *client = pot_client_init("invalid_host_xyz", 9999);
    REQUIRE(client != nullptr);

    // The connection failure should be caught during the first calculation
    double eng = 0;
    double f[3] = {0};
    int32_t atmnrs[1] = {1};
    double pos[3] = {0, 0, 0};
    double box[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    int32_t res = pot_calculate(client, 1, pos, atmnrs, box, &eng, f);
    CHECK(res != 0); // Should fail because host is unreachable

    pot_client_free(client);
  }

  SECTION("Connect and Disconnect repeatedly") {
    // This checks for memory leaks in the init/free cycle
    for (int i = 0; i < 100; ++i) {
      PotClient *client = pot_client_init(HOST.c_str(), PORT);
      if (client) {
        // Just to be sure it's valid
        CHECK(pot_get_last_error(client)[0] == '\0');
        pot_client_free(client);
      } else {
        // If the server isn't running, this acts as a 'connection refused' test
        // don't fail the test, but we stop the loop
        WARN("Server not running at " << HOST << ":" << PORT
                                      << " - skipping lifecycle stress");
        break;
      }
    }
  }
}

TEST_CASE("Bridge Calculation Stress", "[bridge][perf]") {
  PotClient *client = pot_client_init(HOST.c_str(), PORT);
  if (!client) {
    SKIP("Server not available at " << HOST << ":" << PORT);
  }

  // Setup Dummy Data (H2 molecule-ish)
  int32_t natoms = 2;
  std::vector<int32_t> atmnrs = {1, 1};
  std::vector<double> pos = {0.0, 0.0, 0.0, 0.74, 0.0, 0.0};
  std::vector<double> box = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  std::vector<double> forces(natoms * 3);
  double energy = 0;

  // pot_client_init is lazy: connection only opens on first calculate.
  // Without potserv (packaging/unit CI), skip rather than hard-fail unless
  // RGPOT_REQUIRE_POTSERV is set.
  {
    int probe = pot_calculate(client, natoms, pos.data(), atmnrs.data(),
                              box.data(), &energy, forces.data());
    if (probe != 0) {
      std::string err = pot_get_last_error(client);
      pot_client_free(client);
      if (require_potserv()) {
        FAIL("potserv required but not reachable at " << HOST << ":" << PORT
                                                      << ": " << err);
      }
      SKIP("potserv not reachable at " << HOST << ":" << PORT << ": " << err);
    }
  }

  SECTION("Sequential Load (1000 calls)") {
    for (int i = 0; i < 1000; ++i) {
      int res = pot_calculate(client, natoms, pos.data(), atmnrs.data(),
                              box.data(), &energy, forces.data());
      if (res != 0) {
        FAIL("RPC Failed: " << pot_get_last_error(client));
      }
    }
  }

  SECTION("Payload Size Stress (10k atoms)") {
    int32_t big_N = 10000;
    std::vector<int32_t> big_atmnrs(big_N, 1);
    auto big_pos = gen_random_data(big_N * 3);
    std::vector<double> big_forces(big_N * 3);

    int res = pot_calculate(client, big_N, big_pos.data(), big_atmnrs.data(),
                            box.data(), &energy, big_forces.data());

    CHECK(res == 0);
    if (res != 0) {
      UNSCOPED_INFO("Error: " << pot_get_last_error(client));
    }
  }

  pot_client_free(client);
}

TEST_CASE("Bridge Concurrency", "[bridge][threaded]") {
  // Spin up 4 threads, each with its OWN client (recommended usage)
  // Sharing one client across threads requires locking inside the bridge
  // or relying on Cap'n Proto thread-safety which is complex.

  std::atomic<int> success_count{0};
  const int NUM_THREADS = 4;
  const int CALLS_PER_THREAD = 250;

  auto worker = [&]() {
    PotClient *t_client = pot_client_init(HOST.c_str(), PORT);
    if (!t_client)
      return;

    int32_t natoms = 2;
    std::vector<int32_t> atmnrs = {1, 1};
    std::vector<double> pos = {0.0, 0.0, 0.0, 0.74, 0.0, 0.0};
    std::vector<double> box = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    std::vector<double> forces(natoms * 3);
    double energy = 0;

    for (int i = 0; i < CALLS_PER_THREAD; ++i) {
      if (pot_calculate(t_client, natoms, pos.data(), atmnrs.data(), box.data(),
                        &energy, forces.data()) == 0) {
        success_count++;
      }
    }
    pot_client_free(t_client);
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < NUM_THREADS; ++i)
    threads.emplace_back(worker);
  for (auto &t : threads)
    t.join();

  if (success_count > 0 || require_potserv()) {
    CHECK(success_count == NUM_THREADS * CALLS_PER_THREAD);
  } else {
    WARN("Concurrency test skipped (server likely down)");
  }
}

TEST_CASE("Bridge Neighbours", "[bridge][core]") {
  PotNeighborsRequest req{};
  req.cutoff = 2.0;
  req.periodic[0] = 1;
  req.periodic[1] = 1;
  req.periodic[2] = 1;
  req.shifts = 1;
  req.vectors = 1;
  req.distances = 1;
  size_t count = 0;
  size_t needed = 0;
  int32_t atmnrs[2] = {1, 1};
  double pos[6] = {0.0, 0.0, 0.0, 1.2, 0.0, 0.0};
  double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  CHECK(pot_neighbours(nullptr, 2, pos, atmnrs, box, &req, nullptr, &count,
                       &needed) != 0);

  PotClient *client = pot_client_init(HOST.c_str(), PORT);
  if (!client) {
    if (require_potserv()) {
      FAIL("Server required at " << HOST << ":" << PORT);
    }
    SKIP("Server not available at " << HOST << ":" << PORT);
  }
  int probe = pot_neighbours(client, 2, pos, atmnrs, box, &req, nullptr, &count,
                             &needed);
  if (probe < 0) {
    std::string err = pot_get_last_error(client);
    pot_client_free(client);
    if (require_potserv()) {
      FAIL("potserv required but not reachable at " << HOST << ":" << PORT
                                                    << ": " << err);
    }
    SKIP("potserv not reachable at " << HOST << ":" << PORT << ": " << err);
  }
  REQUIRE(probe == 0);
  REQUIRE(needed == 1);
  std::vector<int32_t> first(needed), second(needed), shift(needed * 3);
  std::vector<double> vector(needed * 3), distance(needed);
  std::vector<uint8_t> mask(needed);
  PotNeighborsOut out{};
  out.first = first.data();
  out.second = second.data();
  out.shift = shift.data();
  out.vector = vector.data();
  out.distance = distance.data();
  out.mask = mask.data();
  out.cap = needed;
  REQUIRE(pot_neighbours(client, 2, pos, atmnrs, box, &req, &out, &count,
                         &needed) == 0);
  CHECK(count == 1);
  CHECK(first[0] == 0);
  CHECK(second[0] == 1);
  CHECK(distance[0] == Catch::Approx(1.2));
  pot_client_free(client);
}


