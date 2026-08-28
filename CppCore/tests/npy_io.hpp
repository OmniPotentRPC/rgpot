#pragma once
// Minimal C-order float64 .npy / uncompressed .npz reader for golden masters.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot::testio {

struct NpyArray {
  std::vector<std::size_t> shape;
  std::vector<double> data; // row-major float64
};

inline NpyArray load_npy_stream(std::istream &in, const std::string &label) {
  char magic[6];
  in.read(magic, 6);
  if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("not an npy file: " + label);
  }
  unsigned char ver[2];
  in.read(reinterpret_cast<char *>(ver), 2);
  std::uint32_t hlen = 0;
  if (ver[0] == 1) {
    std::uint16_t hl = 0;
    in.read(reinterpret_cast<char *>(&hl), 2);
    hlen = hl;
  } else {
    in.read(reinterpret_cast<char *>(&hlen), 4);
  }
  std::string header(hlen, '\0');
  in.read(header.data(), hlen);
  if (header.find("'descr': '<f8'") == std::string::npos &&
      header.find("\"descr\": \"<f8\"") == std::string::npos) {
    throw std::runtime_error("npy is not little-endian float64: " + label);
  }
  if (header.find("fortran_order': True") != std::string::npos ||
      header.find("fortran_order\": true") != std::string::npos) {
    throw std::runtime_error("npy is Fortran-order: " + label);
  }
  std::smatch m;
  if (!std::regex_search(header, m, std::regex("shape': \\(([^)]*)\\)"))) {
    throw std::runtime_error("npy header missing shape: " + label);
  }
  NpyArray arr;
  std::stringstream ss(m[1].str());
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    tok.erase(0, tok.find_first_not_of(" \t"));
    if (tok.empty()) {
      continue;
    }
    arr.shape.push_back(static_cast<std::size_t>(std::stoul(tok)));
  }
  std::size_t n = 1;
  for (auto d : arr.shape) {
    n *= d;
  }
  if (arr.shape.empty()) {
    n = 1;
  }
  arr.data.resize(n);
  in.read(reinterpret_cast<char *>(arr.data.data()),
          static_cast<std::streamsize>(n * sizeof(double)));
  if (!in) {
    throw std::runtime_error("npy truncated: " + label);
  }
  return arr;
}

inline NpyArray load_npy(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("missing npy: " + path);
  }
  return load_npy_stream(in, path);
}

inline std::uint64_t read_u16(std::istream &in) {
  std::uint16_t v = 0;
  in.read(reinterpret_cast<char *>(&v), 2);
  return v;
}
inline std::uint64_t read_u32(std::istream &in) {
  std::uint32_t v = 0;
  in.read(reinterpret_cast<char *>(&v), 4);
  return v;
}
inline std::uint64_t read_u64(const char *p) {
  std::uint64_t v = 0;
  std::memcpy(&v, p, 8);
  return v;
}

// Uncompressed ZIP (numpy.savez, ZIP_STORED). Understands ZIP64 extras.
inline std::map<std::string, NpyArray> load_npz(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("missing npz: " + path);
  }
  std::map<std::string, NpyArray> out;
  for (;;) {
    char sig[4];
    in.read(sig, 4);
    if (!in || std::memcmp(sig, "PK\x03\x04", 4) != 0) {
      break;
    }
    (void)read_u16(in); // version
    (void)read_u16(in); // flag
    const auto method = read_u16(in);
    in.seekg(8, std::ios::cur); // time/date/crc
    auto compsize = read_u32(in);
    auto uncomp = read_u32(in);
    const auto namelen = static_cast<std::size_t>(read_u16(in));
    const auto extralen = static_cast<std::size_t>(read_u16(in));
    std::string name(namelen, '\0');
    in.read(name.data(), static_cast<std::streamsize>(namelen));
    std::string extra(extralen, '\0');
    in.read(extra.data(), static_cast<std::streamsize>(extralen));
    if (compsize == 0xffffffffu || uncomp == 0xffffffffu) {
      std::size_t eoff = 0;
      while (eoff + 4 <= extra.size()) {
        std::uint16_t hid = 0, hsz = 0;
        std::memcpy(&hid, extra.data() + eoff, 2);
        std::memcpy(&hsz, extra.data() + eoff + 2, 2);
        eoff += 4;
        if (hid == 0x0001 && eoff + hsz <= extra.size()) {
          std::size_t p = 0;
          if (uncomp == 0xffffffffu && p + 8 <= hsz) {
            uncomp = read_u64(extra.data() + eoff + p);
            p += 8;
          }
          if (compsize == 0xffffffffu && p + 8 <= hsz) {
            compsize = read_u64(extra.data() + eoff + p);
          }
          break;
        }
        eoff += hsz;
      }
    }
    if (method != 0) {
      throw std::runtime_error("npz entry is compressed (use ZIP_STORED): " +
                               path + ":" + name);
    }
    if (compsize > (1ull << 28)) {
      throw std::runtime_error("npz entry implausibly large: " + path + ":" +
                               name);
    }
    std::string payload(static_cast<std::size_t>(compsize), '\0');
    in.read(payload.data(), static_cast<std::streamsize>(compsize));
    if (name.size() >= 4 && name.substr(name.size() - 4) == ".npy") {
      std::string key = name.substr(0, name.size() - 4);
      std::istringstream mem(payload);
      out.emplace(key, load_npy_stream(mem, path + ":" + name));
    }
  }
  if (out.empty()) {
    throw std::runtime_error("npz contained no npy members: " + path);
  }
  return out;
}

} // namespace rgpot::testio
