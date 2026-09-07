#pragma once
// Clear PT_GNU_STACK PF_X inside a stored AOTI .pt2 so dlopen of
// wrapper.so does not need an executable stack (Elja inductor output
// on a kernel that refuses mprotect(PROT_EXEC) on the stack).

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace rgpot {
namespace aoti_execstack {

inline constexpr uint32_t kPtGnuStack = 0x6474e551u;
inline constexpr uint32_t kPfX = 1u;
inline constexpr uint32_t kZipLocal = 0x04034b50u;
inline constexpr uint32_t kZipCentral = 0x02014b50u;

inline uint16_t rd16(const uint8_t *p) {
  return static_cast<uint16_t>(p[0] | (uint16_t(p[1]) << 8));
}

inline uint32_t rd32(const uint8_t *p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}

inline uint64_t rd64(const uint8_t *p) {
  return uint64_t(rd32(p)) | (uint64_t(rd32(p + 4)) << 32);
}

inline uint32_t zip_crc32(const uint8_t *data, size_t n) {
  uint32_t crc = 0xffffffffu;
  for (size_t i = 0; i < n; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit)
      crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}

inline void wr32(uint8_t *p, uint32_t v) {
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
  p[2] = static_cast<uint8_t>(v >> 16);
  p[3] = static_cast<uint8_t>(v >> 24);
}

// Returns true if a GNU_STACK PF_X bit was cleared.
inline bool clear_elf_gnu_stack(uint8_t *data, size_t n) {
  if (n < 64 || data[0] != 0x7f || data[1] != 'E' || data[2] != 'L' ||
      data[3] != 'F')
    return false;
  if (data[4] != 2 || data[5] != 1)
    return false;
  const uint64_t phoff = uint64_t(rd32(data + 32)) |
                         (uint64_t(rd32(data + 36)) << 32);
  const uint16_t phentsize = rd16(data + 54);
  const uint16_t phnum = rd16(data + 56);
  if (phentsize < 8 || phnum == 0)
    return false;
  bool changed = false;
  for (uint16_t i = 0; i < phnum; ++i) {
    const uint64_t off = phoff + uint64_t(i) * phentsize;
    if (off + 8 > n)
      break;
    if (rd32(data + off) != kPtGnuStack)
      continue;
    const uint32_t flags = rd32(data + off + 4);
    if (flags & kPfX) {
      wr32(data + off + 4, flags & ~kPfX);
      changed = true;
    }
  }
  return changed;
}

inline bool elf_needs_gnu_stack_clear(const uint8_t *data, size_t n) {
  if (n < 64 || data[0] != 0x7f || data[1] != 'E' || data[2] != 'L' ||
      data[3] != 'F')
    return false;
  if (data[4] != 2 || data[5] != 1)
    return false;
  const uint64_t phoff = uint64_t(rd32(data + 32)) |
                         (uint64_t(rd32(data + 36)) << 32);
  const uint16_t phentsize = rd16(data + 54);
  const uint16_t phnum = rd16(data + 56);
  if (phentsize < 8 || phnum == 0)
    return false;
  for (uint16_t i = 0; i < phnum; ++i) {
    const uint64_t off = phoff + uint64_t(i) * phentsize;
    if (off + 8 > n)
      break;
    if (rd32(data + off) == kPtGnuStack)
      return (rd32(data + off + 4) & kPfX) != 0;
  }
  return false;
}

// The central directory owns sizes and offsets for ZIP64 and streamed
// records. Every changed payload retains valid ZIP CRC fields.
inline bool scan_or_clear_pt2(uint8_t *buf, size_t n, bool write) {
  auto require_region = [&](uint64_t off, uint64_t len) {
    if (off > n || len > n - off)
      throw std::runtime_error("UmaPot: truncated AOTI ZIP record");
  };
  size_t eocd = n;
  if (n >= 22) {
    const size_t first = n > 65557 ? n - 65557 : 0;
    for (size_t at = n - 22;; --at) {
      if (rd32(buf + at) == 0x06054b50u &&
          size_t(rd16(buf + at + 20)) == n - at - 22) {
        eocd = at;
        break;
      }
      if (at == first)
        break;
    }
  }
  if (eocd == n)
    throw std::runtime_error("UmaPot: AOTI ZIP has no central directory");
  if (rd16(buf + eocd + 4) != 0 || rd16(buf + eocd + 6) != 0)
    throw std::runtime_error("UmaPot: split AOTI ZIP is unsupported");
  uint64_t entries = rd16(buf + eocd + 10);
  uint64_t central_size = rd32(buf + eocd + 12);
  uint64_t central_off = rd32(buf + eocd + 16);
  if (entries == 0xffffu || central_size == 0xffffffffu ||
      central_off == 0xffffffffu) {
    if (eocd < 20 || rd32(buf + eocd - 20) != 0x07064b50u)
      throw std::runtime_error("UmaPot: missing AOTI ZIP64 locator");
    const uint64_t zip64 = rd64(buf + eocd - 12);
    require_region(zip64, 56);
    if (rd32(buf + zip64) != 0x06064b50u ||
        rd32(buf + zip64 + 16) != 0 || rd32(buf + zip64 + 20) != 0)
      throw std::runtime_error("UmaPot: invalid AOTI ZIP64 directory");
    entries = rd64(buf + zip64 + 32);
    central_size = rd64(buf + zip64 + 40);
    central_off = rd64(buf + zip64 + 48);
  }
  require_region(central_off, central_size);
  const size_t central_end = central_off + central_size;
  size_t pos = central_off;
  bool needed = false;
  for (uint64_t entry = 0; entry < entries; ++entry) {
    require_region(pos, 46);
    if (pos > central_end || central_end - pos < 46 ||
        rd32(buf + pos) != kZipCentral)
      throw std::runtime_error("UmaPot: invalid AOTI ZIP central entry");
    const uint16_t flags = rd16(buf + pos + 8);
    const uint16_t method = rd16(buf + pos + 10);
    uint64_t csize = rd32(buf + pos + 20);
    uint64_t usize = rd32(buf + pos + 24);
    const uint16_t namelen = rd16(buf + pos + 28);
    const uint16_t extralen = rd16(buf + pos + 30);
    const uint16_t commentlen = rd16(buf + pos + 32);
    uint64_t local = rd32(buf + pos + 42);
    const size_t name_off = pos + 46;
    const size_t record_len = 46u + namelen + extralen + commentlen;
    if (record_len > central_end - pos)
      throw std::runtime_error("UmaPot: truncated AOTI ZIP central entry");

    if (usize == 0xffffffffu || csize == 0xffffffffu ||
        local == 0xffffffffu) {
      const size_t extra_end = name_off + namelen + extralen;
      size_t extra = name_off + namelen;
      bool found = false;
      while (extra_end - extra >= 4) {
        const uint16_t kind = rd16(buf + extra);
        const uint16_t len = rd16(buf + extra + 2);
        extra += 4;
        if (len > extra_end - extra)
          throw std::runtime_error("UmaPot: truncated AOTI ZIP extra field");
        if (kind == 1) {
          const size_t end = extra + len;
          auto expand = [&](uint64_t &value) {
            if (value != 0xffffffffu)
              return;
            if (end - extra < 8)
              throw std::runtime_error("UmaPot: truncated AOTI ZIP64 size");
            value = rd64(buf + extra);
            extra += 8;
          };
          expand(usize);
          expand(csize);
          expand(local);
          found = true;
          break;
        }
        extra += len;
      }
      if (!found)
        throw std::runtime_error("UmaPot: missing AOTI ZIP64 sizes");
    }

    const bool is_so =
        namelen >= 3 &&
        std::memcmp(buf + name_off + namelen - 3, ".so", 3) == 0;
    if (is_so) {
      if (method != 0 || (flags & 1u))
        throw std::runtime_error("UmaPot: AOTI libraries must use stored ZIP records");
      require_region(local, 30);
      if (rd32(buf + local) != kZipLocal || csize != usize)
        throw std::runtime_error("UmaPot: invalid stored AOTI ZIP library");
      const uint64_t data_off = local + 30 + rd16(buf + local + 26) +
                                rd16(buf + local + 28);
      require_region(data_off, csize);
      if (elf_needs_gnu_stack_clear(buf + data_off, csize)) {
        needed = true;
        if (write) {
          clear_elf_gnu_stack(buf + data_off, csize);
          const uint32_t crc = zip_crc32(buf + data_off, csize);
          wr32(buf + local + 14, crc);
          wr32(buf + pos + 16, crc);
          if (flags & 8u) {
            uint64_t descriptor = data_off + csize;
            require_region(descriptor, 4);
            if (rd32(buf + descriptor) == 0x08074b50u)
              descriptor += 4;
            require_region(descriptor, 4);
            wr32(buf + descriptor, crc);
          }
        }
      }
    }
    pos += record_len;
  }
  return needed;
}

inline std::vector<uint8_t> read_all(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("UmaPot: cannot read AOTI package " + path);
  in.seekg(0, std::ios::end);
  const auto sz = static_cast<size_t>(in.tellg());
  in.seekg(0);
  std::vector<uint8_t> buf(sz);
  if (sz && !in.read(reinterpret_cast<char *>(buf.data()),
                     static_cast<std::streamsize>(sz)))
    throw std::runtime_error("UmaPot: short read of AOTI package " + path);
  return buf;
}

inline void write_all(const std::string &path, const std::vector<uint8_t> &buf) {
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path());
  const std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out)
      throw std::runtime_error("UmaPot: cannot write " + tmp);
    if (!buf.empty() &&
        !out.write(reinterpret_cast<const char *>(buf.data()),
                   static_cast<std::streamsize>(buf.size())))
      throw std::runtime_error("UmaPot: short write of " + tmp);
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec)
    throw std::runtime_error("UmaPot: cannot publish " + path + ": " +
                             ec.message());
}

inline std::string cache_path_for(const std::string &src) {
  namespace fs = std::filesystem;
  const fs::path p(src);
  const auto st = fs::status(p);
  (void)st;
  const auto mtime =
      fs::last_write_time(p).time_since_epoch().count();
  const auto sz = fs::file_size(p);
  const std::string key =
      "zipcd1-" + p.filename().string() + "-" + std::to_string(sz) + "-" +
      std::to_string(static_cast<long long>(mtime));
  fs::path base;
  if (const char *env = std::getenv("RGPOT_AOTI_NOEXEC_DIR"); env && *env)
    base = env;
  else
    base = fs::temp_directory_path() / "rgpot-aoti-noexec";
  return (base / (key + ".pt2")).string();
}

// Path UmaPot should hand to AOTIModelPackageLoader. Original if the
// package already has no executable-stack objects; otherwise a cached
// copy with PT_GNU_STACK PF_X cleared.
inline std::string prepare_pt2_for_load(const std::string &src) {
  auto buf = read_all(src);
  if (!scan_or_clear_pt2(buf.data(), buf.size(), /*write=*/false))
    return src;
  const std::string dst = cache_path_for(src);
  if (std::filesystem::exists(dst) &&
      std::filesystem::file_size(dst) == buf.size()) {
    auto cached = read_all(dst);
    if (!scan_or_clear_pt2(cached.data(), cached.size(), /*write=*/false))
      return dst;
  }
  scan_or_clear_pt2(buf.data(), buf.size(), /*write=*/true);
  write_all(dst, buf);
  return dst;
}

} // namespace aoti_execstack
} // namespace rgpot
