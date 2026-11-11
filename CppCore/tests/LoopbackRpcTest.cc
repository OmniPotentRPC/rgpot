// MIT License
// Copyright 2023--present rgpot developers
//
// The loopback client dials the abstract socket when potserv is listening
// and falls back to TCP when it is not. Results match an in-process Morse.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <capnp/ez-rpc.h>

#include "Potentials.capnp.h"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/abi/Handshake.hpp"
#include "rgpot/rpc/loopback.hpp"
#include "rgpot/rpc/pot_bridge.h"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

std::string sibling_potserv() {
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n < 0)
    return {};
  buf[n] = '\0';
  std::string path(buf);
  const auto slash = path.find_last_of('/');
  if (slash == std::string::npos)
    return "potserv";
  return path.substr(0, slash + 1) + "potserv";
}

int free_port() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  REQUIRE(fd >= 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  REQUIRE(::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0);
  socklen_t len = sizeof(addr);
  REQUIRE(::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len) == 0);
  const int port = ntohs(addr.sin_port);
  ::close(fd);
  return port;
}

class Stub final : public Potential::Server {
public:
  kj::Promise<void> calculate(CalculateContext context) override {
    auto result = context.getResults().initResult();
    result.setEnergy(42.0);
    auto forces = result.initForces(3);
    forces.set(0, 1.0);
    forces.set(1, 2.0);
    forces.set(2, 3.0);
    return kj::READY_NOW;
  }

  kj::Promise<void> getCapabilities(GetCapabilitiesContext context) override {
    auto caps = context.getResults().initCapabilities();
    auto ops = caps.initOperations(2);
    ops.set(0, ::Capabilities::Operation::ENERGY);
    ops.set(1, ::Capabilities::Operation::FORCES);
    rgpot::abi::fill_compatibility(caps);
    return kj::READY_NOW;
  }
};

} // namespace

TEST_CASE("loopback client uses the abstract socket and matches Morse",
          "[rpc][loopback]") {
  const std::string potserv = sibling_potserv();
  const int port = free_port();
  const pid_t child = ::fork();
  REQUIRE(child >= 0);
  if (child == 0) {
    const int devnull = ::open("/dev/null", O_RDWR);
    if (devnull >= 0) {
      ::dup2(devnull, STDOUT_FILENO);
      ::dup2(devnull, STDERR_FILENO);
    }
    const std::string port_text = std::to_string(port);
    ::execl(potserv.c_str(), potserv.c_str(), port_text.c_str(), "Morse",
            nullptr);
    _exit(127);
  }
  struct Reap {
    pid_t pid;
    ~Reap() {
      if (pid > 0) {
        ::kill(pid, SIGTERM);
        ::waitpid(pid, nullptr, 0);
      }
    }
  } reap{child};

  rgpot::MorsePot morse;
  const auto cfg = morse.config();
  const rgpot::types::AtomMatrix positions{{0.0, 0.0, 0.0},
                                           {cfg.re, 0.0, 0.0}};
  const std::vector<int> types{78, 78};
  const std::array<std::array<double, 3>, 3> box{
      {{40.0, 0.0, 0.0}, {0.0, 40.0, 0.0}, {0.0, 0.0, 40.0}}};
  const auto local = morse(positions, types, box);

  const double flat_box[9] = {40, 0, 0, 0, 40, 0, 0, 0, 40};
  const int32_t wire_types[2] = {78, 78};
  double energy = 0.0;
  double forces[6] = {};
  PotClient *client = nullptr;
  bool up = false;
  for (int attempt = 0; attempt < 50 && !up; ++attempt) {
    client = pot_client_init("127.0.0.1", port);
    if (client != nullptr &&
        pot_calculate(client, 2, positions.data(), wire_types, flat_box,
                      &energy, forces) == 0) {
      up = true;
      break;
    }
    pot_client_free(client);
    client = nullptr;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  const char *err = client != nullptr ? pot_get_last_error(client) : "no client";
  INFO(err);
  REQUIRE(up);
  REQUIRE_THAT(energy, WithinRel(std::get<0>(local), 1e-9));
  const auto &local_forces = std::get<1>(local);
  for (int i = 0; i < 6; ++i)
    REQUIRE_THAT(forces[i], WithinAbs(local_forces.data()[i], 1e-9));

  std::ifstream table("/proc/net/unix");
  std::string unix_table((std::istreambuf_iterator<char>(table)),
                         std::istreambuf_iterator<char>());
  REQUIRE(unix_table.find("@rgpot-" + std::to_string(port)) !=
          std::string::npos);
  pot_client_free(client);
}

TEST_CASE("loopback client falls back to TCP when the abstract socket is absent",
          "[rpc][loopback]") {
  const int port = free_port();
  REQUIRE(rgpot::rpc::connect_abstract(port) == -1);

  std::thread server([port] {
    capnp::EzRpcServer rpc(kj::heap<Stub>(), "127.0.0.1", port);
    kj::NEVER_DONE.wait(rpc.getWaitScope());
  });
  server.detach();

  const double pos[3] = {0, 0, 0};
  const int32_t types[1] = {1};
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  double energy = 0.0;
  double forces[3] = {};
  PotClient *client = nullptr;
  bool up = false;
  for (int attempt = 0; attempt < 50 && !up; ++attempt) {
    // Still nobody on the abstract name, including after the TCP server is up.
    REQUIRE(rgpot::rpc::connect_abstract(port) == -1);
    client = pot_client_init("127.0.0.1", port);
    if (client != nullptr &&
        pot_calculate(client, 1, pos, types, box, &energy, forces) == 0) {
      up = true;
      break;
    }
    pot_client_free(client);
    client = nullptr;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  const char *err = client != nullptr ? pot_get_last_error(client) : "no client";
  INFO(err);
  REQUIRE(up);
  REQUIRE(energy == 42.0);
  REQUIRE(forces[0] == 1.0);
  REQUIRE(forces[1] == 2.0);
  REQUIRE(forces[2] == 3.0);
  pot_client_free(client);
}

TEST_CASE("reply poll budget follows the env, the init option, and the setter",
          "[rpc][loopback]") {
  const char *saved = std::getenv("RGPOT_RPC_POLL_US");
  const std::string previous = saved == nullptr ? std::string() : saved;
  const bool had = saved != nullptr;
  unsetenv("RGPOT_RPC_POLL_US");

  PotClient *client = pot_client_init("127.0.0.1", 9);
  REQUIRE(client != nullptr);
  REQUIRE(pot_client_reply_poll_us(client) == 40);
  pot_client_set_reply_poll_us(client, 0);
  REQUIRE(pot_client_reply_poll_us(client) == 0);
  pot_client_set_reply_poll_us(client, 12);
  REQUIRE(pot_client_reply_poll_us(client) == 12);
  pot_client_free(client);

  setenv("RGPOT_RPC_POLL_US", "0", 1);
  client = pot_client_init("127.0.0.1", 9);
  REQUIRE(pot_client_reply_poll_us(client) == 0);
  pot_client_free(client);

  client = pot_client_init_ex("127.0.0.1", 9, 7);
  REQUIRE(pot_client_reply_poll_us(client) == 7);
  pot_client_free(client);

  client = pot_client_init_ex("127.0.0.1", 9, -1);
  REQUIRE(pot_client_reply_poll_us(client) == 0);
  pot_client_free(client);

  if (had)
    setenv("RGPOT_RPC_POLL_US", previous.c_str(), 1);
  else
    unsetenv("RGPOT_RPC_POLL_US");
}
