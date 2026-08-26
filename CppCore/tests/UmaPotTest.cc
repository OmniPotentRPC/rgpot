// MIT License — UmaPot loads an AOTI .pt2, not a metatomic checkpoint.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/UmaPot/UmaPot.hpp"
#include "rgpot/UmaPot/aoti_execstack.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::types::AtomMatrix;
namespace fs = std::filesystem;

// ASE FAIRChemCalculator uma-s-1p1 / omol on Baker 01_hcn reactant.con.
static constexpr double kHcnAseOmolEnergy = -2542.4200325775496;
static constexpr double kHcnAseOmolForces[3][3] = {
    {0.029140297323465347, 0.011646071448922157, -1.4753634929656982},
    {-0.016493169590830803, -0.0065111019648611546, 1.7600582838058472},
    {-0.012647130526602268, -0.0051349685527384281, -0.28469470143318176},
};

static std::string resolve_uma_omol_pt2() {
  if (const char *env = std::getenv("RGPOT_UMA_OMOL_PT2"); env && *env) {
    if (fs::exists(env))
      return std::string(env);
    FAIL("RGPOT_UMA_OMOL_PT2 is set but the file is missing: "
         << env << ". Export an AOTI package with scripts/export_uma_aoti.py");
  }
  // workdir is CppCore/tests; also accept repo-root-relative paths.
  static const char *const kCandidates[] = {
      "data/uma/uma-s-1p1-omol-hcn.pt2",
      "CppCore/tests/data/uma/uma-s-1p1-omol-hcn.pt2",
      "../../bench_data/uma/uma-s-1p1-omol-hcn.pt2",
      "bench_data/uma/uma-s-1p1-omol-hcn.pt2",
  };
  for (const char *path : kCandidates) {
    if (fs::exists(path))
      return std::string(path);
  }
  FAIL("UmaPot HCN fixture needs uma-s-1p1-omol-hcn.pt2. Export with "
       "scripts/export_uma_aoti.py and set RGPOT_UMA_OMOL_PT2, or place the "
       "file at CppCore/tests/data/uma/uma-s-1p1-omol-hcn.pt2 or "
       "bench_data/uma/uma-s-1p1-omol-hcn.pt2");
  return {};
}

static std::vector<uint8_t> elf64_gnu_stack(uint32_t flags) {
  std::vector<uint8_t> elf(120, 0);
  elf[0] = 0x7f;
  elf[1] = 'E';
  elf[2] = 'L';
  elf[3] = 'F';
  elf[4] = 2;
  elf[5] = 1;
  elf[6] = 1;
  elf[16] = 3;
  elf[18] = 62;
  elf[20] = 1;
  elf[32] = 64;
  elf[52] = 64;
  elf[54] = 56;
  elf[56] = 1;
  rgpot::aoti_execstack::wr32(elf.data() + 64, 0x6474e551u);
  rgpot::aoti_execstack::wr32(elf.data() + 68, flags);
  return elf;
}

static std::vector<uint8_t> stored_zip_with_so(const std::vector<uint8_t> &so) {
  const char *name = "model/foo.wrapper.so";
  const uint16_t namelen = static_cast<uint16_t>(std::strlen(name));
  const uint32_t sz = static_cast<uint32_t>(so.size());
  std::vector<uint8_t> z;
  auto push32 = [&](uint32_t v) {
    z.push_back(uint8_t(v));
    z.push_back(uint8_t(v >> 8));
    z.push_back(uint8_t(v >> 16));
    z.push_back(uint8_t(v >> 24));
  };
  auto push16 = [&](uint16_t v) {
    z.push_back(uint8_t(v));
    z.push_back(uint8_t(v >> 8));
  };
  push32(0x04034b50u);
  push16(20);
  push16(0);
  push16(0);
  push16(0);
  push16(0);
  push32(0);
  push32(sz);
  push32(sz);
  push16(namelen);
  push16(0);
  z.insert(z.end(), name, name + namelen);
  z.insert(z.end(), so.begin(), so.end());
  const uint32_t local_len = 30u + namelen + sz;
  push32(0x02014b50u);
  push16(20);
  push16(20);
  push16(0);
  push16(0);
  push16(0);
  push16(0);
  push32(0);
  push32(sz);
  push32(sz);
  push16(namelen);
  push16(0);
  push16(0);
  push16(0);
  push16(0);
  push32(0);
  push32(0);
  z.insert(z.end(), name, name + namelen);
  push32(0x06054b50u);
  push16(0);
  push16(0);
  push16(1);
  push16(1);
  push32(46u + namelen);
  push32(local_len);
  push16(0);
  return z;
}

TEST_CASE("clear_elf_gnu_stack drops PF_X", "[UmaPot][execstack]") {
  auto elf = elf64_gnu_stack(7);
  REQUIRE(rgpot::aoti_execstack::elf_needs_gnu_stack_clear(elf.data(),
                                                           elf.size()));
  REQUIRE(rgpot::aoti_execstack::clear_elf_gnu_stack(elf.data(), elf.size()));
  REQUIRE_FALSE(rgpot::aoti_execstack::elf_needs_gnu_stack_clear(
      elf.data(), elf.size()));
  REQUIRE(rgpot::aoti_execstack::rd32(elf.data() + 68) == 6u);
}

TEST_CASE("prepare_pt2_for_load caches a noexec copy", "[UmaPot][execstack]") {
  auto zip = stored_zip_with_so(elf64_gnu_stack(7));
  const fs::path src = fs::temp_directory_path() / "rgpot-elmc-src.pt2";
  {
    std::ofstream out(src, std::ios::binary | std::ios::trunc);
    REQUIRE(out.write(reinterpret_cast<const char *>(zip.data()),
                      static_cast<std::streamsize>(zip.size())));
  }
  const std::string dst = rgpot::aoti_execstack::prepare_pt2_for_load(src.string());
  REQUIRE(dst != src.string());
  auto patched = rgpot::aoti_execstack::read_all(dst);
  REQUIRE_FALSE(rgpot::aoti_execstack::scan_or_clear_pt2(
      patched.data(), patched.size(), /*write=*/false));
}

TEST_CASE("UmaPot empty model_path throws", "[UmaPot]") {
  rgpot::UmaConfig cfg;
  REQUIRE_THROWS_WITH(rgpot::UmaPot(cfg),
                      Catch::Matchers::ContainsSubstring("model_path"));
}

TEST_CASE("UmaPot rejects a metatomic checkpoint", "[UmaPot]") {
  rgpot::UmaConfig cfg;
  cfg.model_path = "data/lj38/lennard-jones.pt";
  REQUIRE_THROWS_WITH(rgpot::UmaPot(cfg),
                      Catch::Matchers::ContainsSubstring(".pt2"));
}

TEST_CASE("UmaPot paramsKey changes with charge and path", "[UmaPot]") {
  rgpot::UmaConfig cfg;
  cfg.model_path = "no-such-uma.pt2";
  rgpot::UmaPot a(cfg);
  const uint64_t k0 = a.paramsKey();
  a.setChargeSpin(-1, 1);
  REQUIRE(a.paramsKey() != k0);
  cfg.model_path = "other-uma.pt2";
  rgpot::UmaPot b(cfg);
  REQUIRE(b.paramsKey() != k0);
}

TEST_CASE("UmaPot Baker HCN matches ASE FAIRChem omol", "[UmaPot][omol]") {
  const std::string model = resolve_uma_omol_pt2();

  rgpot::UmaConfig cfg;
  cfg.model_path = model;
  cfg.device = "cpu";
  cfg.task_name = "omol";
  cfg.charge = 0;
  cfg.spin = 1;
  rgpot::UmaPot pot(cfg);

  // Baker 01_hcn reactant.con, Angstrom, 25 A cubic cell, PBC.
  const AtomMatrix positions{
      {12.49734736216627162, 12.49892801474515913, 12.54059929828148512},
      {12.50115413363106498, 12.50036504272228832, 11.38209979880783251},
      {12.50149850420264563, 12.50069809648255514, 13.61514544631068446},
  };
  const std::vector<int> atmtypes{6, 7, 1};
  const std::array<std::array<double, 3>, 3> box{
      {{25.0, 0.0, 0.0}, {0.0, 25.0, 0.0}, {0.0, 0.0, 25.0}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  std::printf("backend=UmaPot\n");
  std::printf("model=%s\n", model.c_str());
  std::printf("energy=%.17g\n", energy);
  double max_df = 0.0;
  REQUIRE(forces.rows() == 3);
  REQUIRE(forces.cols() == 3);
  for (size_t i = 0; i < 3; ++i) {
    std::printf("force %zu %.17g %.17g %.17g\n", i, forces(i, 0), forces(i, 1),
                forces(i, 2));
    for (size_t j = 0; j < 3; ++j) {
      max_df =
          std::max(max_df, std::abs(forces(i, j) - kHcnAseOmolForces[i][j]));
    }
  }
  std::printf("max_dF=%.17g\n", max_df);

  REQUIRE_THAT(energy, WithinAbs(kHcnAseOmolEnergy, 1e-4));
  REQUIRE_THAT(max_df, WithinAbs(0.0, 1e-4));
}

TEST_CASE("UmaPot band batch matches per-system evaluation",
          "[UmaPot][omol][band]") {
  // Needs a band package (exported with --batch-max) named beside the
  // single-system fixture or via RGPOT_UMA_OMOL_BAND_PT2. Skipped when
  // absent so the plain fixture suite stays runnable.
  std::string band;
  if (const char *env = std::getenv("RGPOT_UMA_OMOL_BAND_PT2"); env && *env) {
    band = env;
  }
  if (band.empty() || !fs::exists(band)) {
    SKIP("set RGPOT_UMA_OMOL_BAND_PT2 to a --batch-max export");
  }

  rgpot::UmaConfig cfg;
  cfg.model_path = band;
  cfg.device = "cpu";
  cfg.task_name = "omol";
  cfg.charge = 0;
  cfg.spin = 1;
  rgpot::UmaPot pot(cfg);

  // Three HCN geometries: the reference row plus two perturbations.
  const std::array<std::array<double, 9>, 3> geoms{{
      {12.49734736216627162, 12.49892801474515913, 12.54059929828148512,
       12.50115413363106498, 12.50036504272228832, 11.38209979880783251,
       12.50149850420264563, 12.50069809648255514, 13.61514544631068446},
      {12.53734736216627162, 12.49892801474515913, 12.54059929828148512,
       12.50115413363106498, 12.54036504272228832, 11.38209979880783251,
       12.50149850420264563, 12.50069809648255514, 13.57514544631068446},
      {12.45734736216627162, 12.52892801474515913, 12.50059929828148512,
       12.50115413363106498, 12.46036504272228832, 11.42209979880783251,
       12.54149850420264563, 12.50069809648255514, 13.65514544631068446},
  }};
  const std::vector<int> atmtypes{6, 7, 1};
  const std::array<double, 9> box{25.0, 0.0, 0.0, 0.0, 25.0, 0.0,
                                  0.0, 0.0, 25.0};
  const std::array<std::array<double, 3>, 3> box33{
      {{25.0, 0.0, 0.0}, {0.0, 25.0, 0.0}, {0.0, 0.0, 25.0}}};

  // Per-system reference through the public single-call entry (which
  // pads through the band graph itself, so this also covers padding).
  std::array<double, 3> e_single{};
  std::array<std::array<double, 9>, 3> f_single{};
  for (size_t s = 0; s < 3; ++s) {
    AtomMatrix positions(3, 3);
    for (int i = 0; i < 3; ++i)
      for (int d = 0; d < 3; ++d)
        positions(i, d) = geoms[s][static_cast<size_t>(3 * i + d)];
    auto [energy, forces, variance] = pot(positions, atmtypes, box33);
    (void)variance;
    e_single[s] = energy;
    for (int i = 0; i < 3; ++i)
      for (int d = 0; d < 3; ++d)
        f_single[s][static_cast<size_t>(3 * i + d)] = forces(i, d);
  }

  // One batched call over all three through the public batch entry.
  std::vector<rgpot::ForceInput> in;
  std::vector<rgpot::ForceOut> out;
  std::array<std::array<double, 9>, 3> f_batch{};
  for (size_t s = 0; s < 3; ++s) {
    in.push_back(rgpot::ForceInput{3, geoms[s].data(), atmtypes.data(),
                                   box.data()});
    out.push_back(rgpot::ForceOut{f_batch[s].data(), 0.0, 0.0});
  }
  const rgpot::ForceBatch batch{3, in.data(), out.data()};
  pot.forceBatch(batch);

  for (size_t s = 0; s < 3; ++s) {
    REQUIRE_THAT(out[s].energy, WithinAbs(e_single[s], 1e-8));
    for (size_t k = 0; k < 9; ++k) {
      REQUIRE_THAT(f_batch[s][k], WithinAbs(f_single[s][k], 1e-7));
    }
  }
}
