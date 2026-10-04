// SPDX-License-Identifier: MIT
#include "rgpot/UmaPot/aoti_execstack.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <iostream>
#include <thread>

int main() {
  namespace fs = std::filesystem;
  const auto root =
      fs::temp_directory_path() /
      ("rgpot-publication-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto target = root / "model.pt2";
  constexpr int writers = 8;
  constexpr int iterations = 8;
  constexpr std::size_t size = 4 * 1024 * 1024;
  std::atomic<int> failures{0};
  std::atomic<int> finished{0};
  std::atomic<int> reads{0};
  std::barrier start(writers + 1);
  std::vector<std::string> errors(writers);
  rgpot::aoti_execstack::write_all(target.string(),
                                   std::vector<uint8_t>(size, 0));
  std::vector<std::thread> threads;
  for (int writer = 0; writer < writers; ++writer) {
    threads.emplace_back([&, writer] {
      const std::vector<uint8_t> payload(size,
                                         static_cast<uint8_t>(writer + 1));
      start.arrive_and_wait();
      try {
        for (int iteration = 0; iteration < iterations; ++iteration)
          rgpot::aoti_execstack::write_all(target.string(), payload);
      } catch (const std::exception &error) {
        errors[writer] = error.what();
        ++failures;
      }
      ++finished;
    });
  }
  start.arrive_and_wait();
  while (finished < writers) {
    try {
      const auto data = rgpot::aoti_execstack::read_all(target.string());
      if (data.size() != size ||
          !std::all_of(data.begin(), data.end(),
                       [&](uint8_t value) { return value == data.front(); })) {
        std::cerr << "Reader observed a partial or mixed publication\n";
        ++failures;
      }
      ++reads;
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      ++failures;
    }
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &error : errors)
    if (!error.empty())
      std::cerr << error << '\n';
  if (reads == 0)
    ++failures;

  const auto blocked = root / "blocked.pt2";
  fs::create_directory(blocked);
  bool refused = false;
  try {
    rgpot::aoti_execstack::write_all(blocked.string(), {1, 2, 3});
  } catch (const std::exception &) {
    refused = true;
  }
  if (!refused || !fs::is_directory(blocked))
    ++failures;
  for (const auto &entry : fs::directory_iterator(root)) {
    if (entry.path() != target && entry.path() != blocked) {
      std::cerr << "Unowned temporary remains: " << entry.path() << '\n';
      ++failures;
    }
  }
  std::cout << "writers=" << writers << " publications=" << writers * iterations
            << " reads=" << reads << " failures=" << failures << '\n';
  fs::remove_all(root);
  return failures == 0 ? 0 : 1;
}
