#pragma once
// Minimal .npy / .npz (ZIP of .npy) reader for float64 C-order goldens.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <zlib.h>

namespace rgpot {
namespace testio {

struct NpyArray {
  std::vector<std::size_t> shape;
  std::vector<double> data; // C-order float64
};

inline void require(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

inline NpyArray parse_npy_bytes(const std::vector<unsigned char> &buf,
                                const std::string &label) {
  require(buf.size() >= 10, label + ": truncated npy");
  require(std::memcmp(buf.data(), "\x93NUMPY", 6) == 0, label + ": not npy");
  const unsigned char major = buf[6];
  require(major == 1 || major == 2, label + ": unsupported npy version");
  std::size_t header_len = 0;
  std::size_t data_off = 0;
  if (major == 1) {
    header_len = static_cast<std::size_t>(buf[8]) |
                 (static_cast<std::size_t>(buf[9]) << 8);
    data_off = 10 + header_len;
  } else {
    require(buf.size() >= 12, label + ": truncated npy v2");
    header_len = static_cast<std::size_t>(buf[8]) |
                 (static_cast<std::size_t>(buf[9]) << 8) |
                 (static_cast<std::size_t>(buf[10]) << 16) |
                 (static_cast<std::size_t>(buf[11]) << 24);
    data_off = 12 + header_len;
  }
  require(data_off <= buf.size(), label + ": header overruns file");
  const std::string header(reinterpret_cast<const char *>(buf.data() +
                                                          (major == 1 ? 10 : 12)),
                           header_len);
  require(header.find("'descr': '<f8'") != std::string::npos ||
              header.find("\"descr\": \"<f8\"") != std::string::npos ||
              header.find("'descr': '<f8'") != std::string::npos,
          label + ": expected little-endian float64");
  require(header.find("'fortran_order': False") != std::string::npos ||
              header.find("'fortran_order': False") != std::string::npos ||
              header.find("\"fortran_order\": False") != std::string::npos,
          label + ": expected C-order");
  const auto shape_pos = header.find("shape");
  require(shape_pos != std::string::npos, label + ": no shape");
  const auto lpar = header.find('(', shape_pos);
  const auto rpar = header.find(')', lpar);
  require(lpar != std::string::npos && rpar != std::string::npos,
          label + ": bad shape");
  NpyArray arr;
  std::string nums = header.substr(lpar + 1, rpar - lpar - 1);
  std::stringstream ss(nums);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    // trim
    auto a = tok.find_first_not_of(" \t\r\n");
    auto b = tok.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) {
      continue;
    }
    tok = tok.substr(a, b - a + 1);
    if (tok.empty()) {
      continue;
    }
    arr.shape.push_back(static_cast<std::size_t>(std::stoull(tok)));
  }
  if (arr.shape.empty()) {
    arr.shape.push_back(1); // numpy scalar
  }
  std::size_t n = 1;
  for (auto d : arr.shape) {
    n *= d;
  }
  require(data_off + n * sizeof(double) <= buf.size(),
          label + ": data shorter than shape");
  arr.data.resize(n);
  std::memcpy(arr.data.data(), buf.data() + data_off, n * sizeof(double));
  return arr;
}

inline NpyArray load_npy(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  require(static_cast<bool>(in), "missing golden fixture: " + path);
  in.seekg(0, std::ios::end);
  const auto sz = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  std::vector<unsigned char> buf(sz);
  in.read(reinterpret_cast<char *>(buf.data()),
          static_cast<std::streamsize>(sz));
  require(static_cast<bool>(in), "failed to read " + path);
  return parse_npy_bytes(buf, path);
}

inline std::vector<unsigned char>
inflate_raw(const unsigned char *src, std::size_t nsrc, std::size_t ndst,
            const std::string &label) {
  std::vector<unsigned char> out(ndst);
  z_stream strm{};
  strm.next_in = const_cast<Bytef *>(reinterpret_cast<const Bytef *>(src));
  strm.avail_in = static_cast<uInt>(nsrc);
  strm.next_out = reinterpret_cast<Bytef *>(out.data());
  strm.avail_out = static_cast<uInt>(ndst);
  require(inflateInit2(&strm, -MAX_WBITS) == Z_OK, label + ": inflateInit");
  const int rc = inflate(&strm, Z_FINISH);
  inflateEnd(&strm);
  require(rc == Z_STREAM_END, label + ": inflate failed");
  return out;
}

inline std::uint16_t u16(const unsigned char *p) {
  return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}
inline std::uint32_t u32(const unsigned char *p) {
  return static_cast<std::uint32_t>(p[0] | (p[1] << 8) | (p[2] << 16) |
                                    (p[3] << 24));
}

inline std::unordered_map<std::string, NpyArray>
load_npz(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  require(static_cast<bool>(in), "missing golden fixture: " + path);
  in.seekg(0, std::ios::end);
  const auto sz = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  std::vector<unsigned char> buf(sz);
  in.read(reinterpret_cast<char *>(buf.data()),
          static_cast<std::streamsize>(sz));
  require(static_cast<bool>(in), "failed to read " + path);
  require(sz >= 22, path + ": too small for zip");

  // Locate EOCD (numpy npz uses a short comment-less directory).
  std::size_t eocd = sz - 22;
  bool found = false;
  for (std::size_t i = 0; i < 65536 && i + 22 <= sz; ++i) {
    const std::size_t cand = sz - 22 - i;
    if (std::memcmp(buf.data() + cand, "PK\x05\x06", 4) == 0) {
      eocd = cand;
      found = true;
      break;
    }
  }
  require(found, path + ": no zip EOCD");
  const std::uint16_t nent = u16(buf.data() + eocd + 10);
  const std::uint32_t cd_off = u32(buf.data() + eocd + 16);

  std::unordered_map<std::string, NpyArray> out;
  std::size_t off = cd_off;
  for (std::uint16_t i = 0; i < nent; ++i) {
    require(off + 46 <= sz, path + ": central dir overruns file");
    require(std::memcmp(buf.data() + off, "PK\x01\x02", 4) == 0,
            path + ": bad central dir signature");
    const unsigned method = u16(buf.data() + off + 10);
    const std::uint32_t comp_size = u32(buf.data() + off + 20);
    const std::uint32_t uncomp_size = u32(buf.data() + off + 24);
    const unsigned name_len = u16(buf.data() + off + 28);
    const unsigned extra_len = u16(buf.data() + off + 30);
    const unsigned comment_len = u16(buf.data() + off + 32);
    const std::uint32_t local_off = u32(buf.data() + off + 42);
    require(off + 46 + name_len + extra_len + comment_len <= sz,
            path + ": central dir name overruns");
    std::string fname(reinterpret_cast<const char *>(buf.data() + off + 46),
                      name_len);
    require(local_off + 30 <= sz, path + ": local header offset past EOF");
    require(std::memcmp(buf.data() + local_off, "PK\x03\x04", 4) == 0,
            path + ": bad local header");
    const unsigned loc_name = u16(buf.data() + local_off + 26);
    const unsigned loc_extra = u16(buf.data() + local_off + 28);
    const std::size_t payload_off = local_off + 30 + loc_name + loc_extra;
    require(payload_off + comp_size <= sz, path + ": payload overruns file");
    const unsigned char *payload = buf.data() + payload_off;
    std::vector<unsigned char> npy;
    if (method == 0) {
      npy.assign(payload, payload + uncomp_size);
    } else if (method == 8) {
      npy = inflate_raw(payload, comp_size, uncomp_size, path + ":" + fname);
    } else {
      throw std::runtime_error(path + ": unsupported zip method for " + fname);
    }
    if (fname.size() >= 4 && fname.substr(fname.size() - 4) == ".npy") {
      fname = fname.substr(0, fname.size() - 4);
    }
    out.emplace(fname, parse_npy_bytes(npy, path + ":" + fname));
    off += 46 + name_len + extra_len + comment_len;
  }
  require(!out.empty(), path + ": no npy members");
  return out;
}

} // namespace testio
} // namespace rgpot
