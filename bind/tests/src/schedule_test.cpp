// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file schedule_test.cpp
 * @author rgpot Developers
 * @date 2026-10-11
 * @brief Pure C++ check that the F5 schedule ignores the worker count.
 *
 * This translation unit defines RGBIND_NO_PYTHON. It does not initialise
 * an interpreter and does not link nanobind.
 */

#ifndef RGBIND_NO_PYTHON
#define RGBIND_NO_PYTHON
#endif
#include "rgbind/rgbind.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void sha256(const std::uint8_t *message, std::size_t len, std::uint8_t out[32]) {
  static const std::uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
      0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  std::uint32_t h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372,
                h3 = 0xa54ff53a, h4 = 0x510e527f, h5 = 0x9b05688c,
                h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
  const std::uint64_t bits = static_cast<std::uint64_t>(len) * 8;
  const std::size_t padded = ((len + 9 + 63) / 64) * 64;
  std::vector<std::uint8_t> buf(padded, 0);
  if (len != 0) {
    std::memcpy(buf.data(), message, len);
  }
  buf[len] = 0x80;
  for (int i = 0; i < 8; ++i) {
    buf[padded - 1 - static_cast<std::size_t>(i)] =
        static_cast<std::uint8_t>((bits >> (8 * i)) & 0xff);
  }
  auto rotr = [](std::uint32_t x, std::uint32_t n) {
    return (x >> n) | (x << (32 - n));
  };
  for (std::size_t off = 0; off < padded; off += 64) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<std::uint32_t>(buf[off + 4 * i]) << 24) |
             (static_cast<std::uint32_t>(buf[off + 4 * i + 1]) << 16) |
             (static_cast<std::uint32_t>(buf[off + 4 * i + 2]) << 8) |
             static_cast<std::uint32_t>(buf[off + 4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 =
          rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 =
          rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6,
                  h = h7;
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t t1 = h + S1 + ch + k[i] + w[i];
      const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = S0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;
    h4 += e;
    h5 += f;
    h6 += g;
    h7 += h;
  }
  const std::uint32_t hs[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
  for (int i = 0; i < 8; ++i) {
    out[4 * i] = static_cast<std::uint8_t>((hs[i] >> 24) & 0xff);
    out[4 * i + 1] = static_cast<std::uint8_t>((hs[i] >> 16) & 0xff);
    out[4 * i + 2] = static_cast<std::uint8_t>((hs[i] >> 8) & 0xff);
    out[4 * i + 3] = static_cast<std::uint8_t>(hs[i] & 0xff);
  }
}

std::string hex32(const std::uint8_t dig[32]) {
  static const char *kHex = "0123456789abcdef";
  std::string text(64, '0');
  for (int i = 0; i < 32; ++i) {
    text[2 * i] = kHex[dig[i] >> 4];
    text[2 * i + 1] = kHex[dig[i] & 0xf];
  }
  return text;
}

std::string hash_text(const std::string &text) {
  std::uint8_t dig[32];
  sha256(reinterpret_cast<const std::uint8_t *>(text.data()), text.size(), dig);
  return hex32(dig);
}

std::vector<rgbind::OwnerRow> fixture() {
  const std::int64_t spec[][3] = {
      {0, 10, 0}, {2, 10, 0}, {1, 20, 1}, {4, 10, 0}, {3, 30, 0}, {5, 20, 1},
  };
  std::vector<rgbind::OwnerRow> rows;
  for (const auto &item : spec) {
    rgbind::OwnerRow row;
    row.row_id = item[0];
    row.owner = static_cast<std::uint64_t>(item[1]);
    row.reentrant = item[2] != 0;
    rows.push_back(row);
  }
  return rows;
}

} // namespace

int main() {
  const std::string empty_hash = hash_text("");
  const std::string abc_hash = hash_text("abc");
  if (empty_hash !=
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") {
    std::cerr << "sha256 empty mismatch " << empty_hash << "\n";
    return 1;
  }
  if (abc_hash !=
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
    std::cerr << "sha256 abc mismatch " << abc_hash << "\n";
    return 1;
  }

  const std::vector<rgbind::OwnerRow> rows = fixture();
  const int workers[] = {1, 2, 3, 8};
  std::string first;
  for (int n : workers) {
    const std::string canonical = rgbind::serialise_by_owner(rows, n).canonical();
    const std::string dig = hash_text(canonical);
    std::cout << "schedule_sha256 workers=" << n << " " << dig << "\n";
    if (first.empty()) {
      first = dig;
    } else if (dig != first) {
      std::cerr << "schedule hash changed at " << n << " workers\n";
      return 1;
    }
  }
  try {
    (void)rgbind::serialise_by_owner(rows, 0);
    std::cerr << "n_workers=0 did not throw\n";
    return 1;
  } catch (const std::invalid_argument &) {
  }

  const std::string base = rgbind::serialise_by_owner(rows, 1).canonical();
  std::vector<rgbind::OwnerRow> reversed = rows;
  std::reverse(reversed.begin(), reversed.end());
  std::vector<rgbind::OwnerRow> rotated = rows;
  std::rotate(rotated.begin(), rotated.begin() + 2, rotated.end());
  std::vector<rgbind::OwnerRow> swapped = rows;
  std::swap(swapped.front(), swapped.back());
  const std::vector<rgbind::OwnerRow> *perms[] = {&rows, &reversed, &rotated, &swapped};
  const char *names[] = {"identity", "reverse", "rotate-left-2", "swap-ends"};
  for (int i = 0; i < 4; ++i) {
    const std::string canonical = rgbind::serialise_by_owner(*perms[i], 3).canonical();
    const std::string dig = hash_text(canonical);
    std::cout << "schedule_sha256 permutation=" << names[i] << " " << dig << "\n";
    if (canonical != base || dig != first) {
      std::cerr << "permutation " << names[i] << " changed the schedule\n";
      return 1;
    }
  }

  std::vector<rgbind::OwnerRow> duplicated = rows;
  duplicated.push_back(rows.front());
  try {
    (void)rgbind::serialise_by_owner(duplicated, 1);
    std::cerr << "duplicate row_id did not throw\n";
    return 1;
  } catch (const std::invalid_argument &err) {
    const std::string text = err.what();
    if (text.find("duplicate row_id") == std::string::npos) {
      std::cerr << "duplicate row_id message was " << text << "\n";
      return 1;
    }
  }

  std::cout << "identical=1\n";
  return 0;
}
