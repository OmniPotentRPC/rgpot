// MIT License
// Copyright 2023--present rgpot developers
//
// Wall time per force call through the Cap'n Proto RPC path: the C client
// bridge (pot_bridge.h) against a running potserv, on the Pt nanoparticles
// of time_pair_scaling. Not a meson test. Start the server first:
//   potserv 12345 Morse &
//   time_rpc_call --port 12345 --n 1000 --calls 200
// The difference from time_pair_scaling's in-process figure for the same
// system and mode is the serialisation and transport cost per call.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bench_systems.hpp"
#include "rgpot/rpc/pot_bridge.h"

int main(int argc, char **argv) {
  std::string host = "localhost";
  int port = 12345;
  std::size_t n = 1000;
  long calls = 200;
  int repeats = 5;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--host") == 0)
      host = argv[i + 1];
    else if (std::strcmp(argv[i], "--port") == 0)
      port = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--n") == 0)
      n = std::strtoull(argv[i + 1], nullptr, 10);
    else if (std::strcmp(argv[i], "--calls") == 0)
      calls = std::atol(argv[i + 1]);
    else if (std::strcmp(argv[i], "--repeats") == 0)
      repeats = std::atoi(argv[i + 1]);
  }
  PotClient *client = pot_client_init(host.c_str(), port);
  if (client == nullptr) {
    std::fprintf(stderr, "cannot connect to %s:%d\n", host.c_str(), port);
    return 1;
  }
  const auto s = rgpot_bench::nanoparticle(n, 3.92, 10.5, 78);
  const auto natoms = static_cast<int32_t>(s.types.size());
  std::vector<int32_t> types(s.types.begin(), s.types.end());
  std::vector<double> F(s.pos.size(), 0.0);
  double energy = 0.0;

  std::vector<double> perCall;
  for (int r = 0; r < repeats; ++r) {
    // Same geometry every call: the server's pair list hits, so the
    // figure is the transport plus the warm kernel.
    for (int w = 0; w < 3; ++w) {
      if (pot_calculate(client, natoms, s.pos.data(), types.data(),
                        s.box.data(), &energy, F.data()) != 0) {
        std::fprintf(stderr, "pot_calculate: %s\n",
                     pot_get_last_error(client));
        pot_client_free(client);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (long c = 0; c < calls; ++c) {
      pot_calculate(client, natoms, s.pos.data(), types.data(), s.box.data(),
                    &energy, F.data());
    }
    const auto t1 = std::chrono::steady_clock::now();
    perCall.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count() /
        static_cast<double>(calls));
  }
  pot_client_free(client);
  std::sort(perCall.begin(), perCall.end());
  std::printf("system,mode,natoms,calls,repeats,us_per_call_median,us_per_"
              "call_min,us_per_call_max,energy_last\n");
  std::printf("pt-np-rpc,warm,%d,%ld,%d,%.3f,%.3f,%.3f,%.12g\n", natoms, calls,
              repeats, perCall[perCall.size() / 2], perCall.front(),
              perCall.back(), energy);
  return 0;
}
