// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_all.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/rpc/pot_bridge.h"

namespace {

bool sameBits(double a, double b) {
  std::uint64_t x = 0;
  std::uint64_t y = 0;
  std::memcpy(&x, &a, sizeof(x));
  std::memcpy(&y, &b, sizeof(y));
  return x == y;
}

pid_t spawn(const char *potserv, const char *address) {
  const pid_t pid = fork();
  if (pid == 0) {
    execl(potserv, potserv, address, "Morse", static_cast<char *>(nullptr));
    _exit(127);
  }
  return pid;
}

void stop(pid_t pid) {
  if (pid > 0) {
    kill(pid, SIGTERM);
    waitpid(pid, nullptr, 0);
  }
}

template <typename Connect>
bool waitReady(Connect connect) {
  for (int i = 0; i < 100; ++i) {
    PotClient *client = connect();
    if (client != nullptr) {
      double energy = 0.0;
      double forces[6] = {};
      const double pos[6] = {0, 0, 0, 2.7, 0, 0};
      const int types[2] = {78, 78};
      const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
      const int rc = pot_calculate(client, 2, pos, types, box, &energy, forces);
      pot_client_free(client);
      if (rc == 0) {
        return true;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return false;
}

} // namespace

TEST_CASE("a unix socket call matches an in-process Morse evaluation", "[rpc][unix]") {
  const char *potserv = std::getenv("RGPOT_POTSERV");
  REQUIRE(potserv != nullptr);
  const std::string path = "/tmp/rgpot-unix-" + std::to_string(getpid()) + ".sock";
  const std::string address = "unix:" + path;
  ::unlink(path.c_str());
  const pid_t pid = spawn(potserv, address.c_str());
  REQUIRE(pid > 0);

  REQUIRE(waitReady([&]() { return pot_client_init(address.c_str(), 0); }));

  const double pos[6] = {0.1, 0.2, 0.3, 2.8, 0.4, -0.2};
  const int types[2] = {78, 78};
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  double rpc_energy = 0.0;
  double rpc_forces[6] = {};
  PotClient *client = pot_client_init(address.c_str(), 0);
  REQUIRE(client != nullptr);
  REQUIRE(pot_calculate(client, 2, pos, types, box, &rpc_energy, rpc_forces) == 0);
  pot_client_free(client);

  rgpot::MorsePot pot;
  std::vector<double> forces(6);
  rgpot::ForceInput input{2, pos, types, box};
  rgpot::ForceOut out{forces.data(), 0, 0, {}, 0};
  pot.forceImpl(input, &out);
  REQUIRE(sameBits(rpc_energy, out.energy));
  for (int i = 0; i < 6; ++i) {
    REQUIRE(sameBits(rpc_forces[i], forces[static_cast<std::size_t>(i)]));
  }

  // A numeric port still reaches the same server binary.
  const pid_t tcp = spawn(potserv, "23995");
  REQUIRE(tcp > 0);
  REQUIRE(waitReady([]() { return pot_client_init("127.0.0.1", 23995); }));
  PotClient *tcp_client = pot_client_init("127.0.0.1", 23995);
  REQUIRE(tcp_client != nullptr);
  double tcp_energy = 0.0;
  double tcp_forces[6] = {};
  REQUIRE(pot_calculate(tcp_client, 2, pos, types, box, &tcp_energy, tcp_forces) == 0);
  REQUIRE(sameBits(tcp_energy, out.energy));
  pot_client_free(tcp_client);

  stop(pid);
  stop(tcp);
  ::unlink(path.c_str());
}

TEST_CASE("potserv leaves a regular file in place", "[rpc][unix]") {
  const char *potserv = std::getenv("RGPOT_POTSERV");
  REQUIRE(potserv != nullptr);
  const std::string path = "/tmp/rgpot-unix-file-" + std::to_string(getpid());
  const std::string address = "unix:" + path;
  {
    const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    REQUIRE(fd >= 0);
    const char text[] = "keep";
    REQUIRE(::write(fd, text, 4) == 4);
    ::close(fd);
  }
  const pid_t pid = spawn(potserv, address.c_str());
  REQUIRE(pid > 0);
  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) != 0);
  struct stat st;
  REQUIRE(lstat(path.c_str(), &st) == 0);
  REQUIRE(S_ISREG(st.st_mode));
  char buf[8] = {};
  const int fd = ::open(path.c_str(), O_RDONLY);
  REQUIRE(fd >= 0);
  REQUIRE(::read(fd, buf, sizeof(buf)) == 4);
  ::close(fd);
  REQUIRE(std::string(buf, 4) == "keep");
  ::unlink(path.c_str());
}
