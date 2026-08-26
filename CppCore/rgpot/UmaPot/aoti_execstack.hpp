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

// Walk stored zip local-file payloads. Returns true if any ELF needed
// a GNU_STACK clear. Throws if a deflated .so still needs one.
inline bool scan_or_clear_pt2(uint8_t *buf, size_t n, bool write) {
  bool needed = false;
  size_t pos = 0;
  while (pos + 30 <= n) {
    const uint32_t sig = rd32(buf + pos);
    if (sig == kZipCentral)
      break;
    if (sig != kZipLocal)
      break;
    const uint16_t flags = rd16(buf + pos + 6);
    const uint16_t method = rd16(buf + pos + 8);
    const uint32_t csize = rd32(buf + pos + 18);
    const uint16_t namelen = rd16(buf + pos + 26);
    const uint16_t extralen = rd16(buf + pos + 28);
    const size_t name_off = pos + 30;
    if (name_off + namelen + extralen > n)
      break;
    const size_t data_off = name_off + namelen + extralen;
    if (data_off + csize > n)
      break;
    const bool is_so =
        namelen >= 3 &&
        std::memcmp(buf + name_off + namelen - 3, ".so", 3) == 0;
    if (is_so) {
      if (method != 0) {
        if (elf_needs_gnu_stack_clear(buf + data_off, csize))
          throw std::runtime_error(
              "UmaPot: AOTI wrapper.so needs an executable stack and is "
              "deflated; remint on this host or rewrite the package with "
              "scripts/export_uma_aoti.py");
      } else if (write) {
        needed = clear_elf_gnu_stack(buf + data_off, csize) || needed;
      } else {
        needed = elf_needs_gnu_stack_clear(buf + data_off, csize) || needed;
      }
    }
    (void)flags;
    pos = data_off + csize;
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
      p.filename().string() + "-" + std::to_string(sz) + "-" +
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
