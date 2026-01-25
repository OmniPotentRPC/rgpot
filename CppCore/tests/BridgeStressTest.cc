// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
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

  // If server is up, expect all calls to succeed
  // If server is down, count is 0, which is also a valid state for the test
  // code (don't crash) but ideally we want to see successes.
  if (success_count > 0) {
    CHECK(success_count == NUM_THREADS * CALLS_PER_THREAD);
  } else {
    WARN("Concurrency test skipped (server likely down)");
  }
}
