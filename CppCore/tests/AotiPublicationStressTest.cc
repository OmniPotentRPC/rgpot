// SPDX-License-Identifier: MIT
#include "rgpot/UmaPot/aoti_execstack.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kWriters = 8;
constexpr int kReaders = 4;
constexpr int kIterations = 8;
constexpr int kRounds = 20;
constexpr std::size_t kSize = 256 * 1024;

int run_round(const std::filesystem::path &target, std::string &error) {
  std::atomic<int> failures{0};
  std::atomic<int> finished{0};
  std::atomic<int> reads{0};
  std::barrier start(kWriters + kReaders);
  std::mutex error_mu;
  std::vector<std::string> writer_errors(static_cast<std::size_t>(kWriters));
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(kWriters + kReaders));

  auto note = [&](const std::string &text) {
    std::lock_guard<std::mutex> lock(error_mu);
    if (error.empty())
      error = text;
  };

  for (int writer = 0; writer < kWriters; ++writer) {
    threads.emplace_back([&, writer] {
      const std::vector<uint8_t> payload(kSize,
                                         static_cast<uint8_t>(writer + 1));
      start.arrive_and_wait();
      try {
        for (int iteration = 0; iteration < kIterations; ++iteration)
          rgpot::aoti_execstack::write_all(target.string(), payload);
      } catch (const std::exception &ex) {
        writer_errors[static_cast<std::size_t>(writer)] = ex.what();
        ++failures;
      }
      ++finished;
    });
  }

  for (int reader = 0; reader < kReaders; ++reader) {
    threads.emplace_back([&] {
      start.arrive_and_wait();
      while (finished.load() < kWriters) {
        try {
          const auto data = rgpot::aoti_execstack::read_all(target.string());
          if (data.size() != kSize ||
              !std::all_of(data.begin(), data.end(), [&](uint8_t value) {
                return value == data.front();
              })) {
            note("Reader observed a partial or mixed publication");
            ++failures;
          }
          ++reads;
        } catch (const std::exception &ex) {
          note(ex.what());
          ++failures;
        }
      }
    });
  }

  for (auto &thread : threads)
    thread.join();

  for (const auto &writer_error : writer_errors) {
    if (!writer_error.empty())
      note(writer_error);
  }
  if (reads.load() == 0) {
    note("Reader observed no publication");
    ++failures;
  }
  std::cout << "writers=" << kWriters
            << " publications=" << kWriters * kIterations
            << " reads=" << reads.load() << " failures=" << failures.load()
            << '\n';
  return failures.load() == 0 ? 0 : 1;
}

} // namespace

int main() {
  namespace fs = std::filesystem;
  const auto root =
      fs::temp_directory_path() /
      ("rgpot-publication-stress-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(root);
  const auto target = root / "model.pt2";
  rgpot::aoti_execstack::write_all(target.string(),
                                   std::vector<uint8_t>(kSize, 0));

  int failed_rounds = 0;
  for (int round = 1; round <= kRounds; ++round) {
    std::string error;
    const int status = run_round(target, error);
    if (status != 0) {
      ++failed_rounds;
      if (!error.empty())
        std::cerr << error << '\n';
    }
    std::cout << "round=" << round << " failures=" << status << '\n';
    std::cout.flush();
  }
  std::cout << "rounds=" << kRounds << " failed_rounds=" << failed_rounds
            << '\n';
  fs::remove_all(root);
  return failed_rounds == 0 ? 0 : 1;
}
