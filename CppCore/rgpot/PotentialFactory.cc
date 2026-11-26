// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/PotentialFactory.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/exception.h>

#include "rgpot/EnginePot/EnginePot.hpp"
#include "rgpot/ExternalPot/ExternalPot.hpp"
#include "rgpot/MOPACPot/MOPACPot.hpp"
#include "rgpot/NWChemPot/NWChemPot.hpp"
#include "rgpot/ORCACPot/ORCACPot.hpp"
#include "rgpot/abi/EngineDlopen.hpp"
#include "rgpot/rpc/CompositeConfigure.hpp"
#include "rgpot/rpc/D2Configure.hpp"
#include "rgpot/rpc/MorseConfigure.hpp"
#include "rgpot/rpc/MullerBrownConfigure.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#ifdef RGPOT_HAS_DFTD3
#include "rgpot/D3Pot/D3Pot.hpp"
#endif
#ifdef RGPOT_HAS_DFTD4
#include "rgpot/rpc/D4Configure.hpp"
#endif
#ifdef RGPOT_HAS_XTB
#include "rgpot/XTBPot/XTBPot.hpp"
#endif
#ifdef RGPOT_HAS_TBLITE
#include "rgpot/TBLitePot/TBLitePot.hpp"
#endif
#ifdef RGPOT_HAS_SKALA
#include "rgpot/SkalaPot/SkalaPot.hpp"
#endif
#ifdef RGPOT_HAS_CP2K
#include "rgpot/CP2KPot/CP2KPot.hpp"
#endif
#ifdef RGPOT_HAS_EXPR
#include "rgpot/ExprPot/ExprPot.hpp"
#endif

namespace rgpot {
namespace {

std::string lowered(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return s;
}

[[noreturn]] void notBuilt(const std::string &arm, const std::string &option) {
  throw std::invalid_argument("makePotential: the " + arm +
                              " arm needs a build with " + option +
                              ", which this rgpot was not configured with");
}

/// A word-aligned flat-array message whose root is @p root.
template <typename Reader> std::vector<capnp::word> flatMessage(Reader root) {
  capnp::MallocMessageBuilder message;
  message.setRoot(root);
  const auto words = capnp::messageToFlatArray(message);
  return std::vector<capnp::word>(words.begin(), words.end());
}

std::unique_ptr<PotentialBase>
enginePot(const std::vector<std::string> &candidates,
          const std::vector<capnp::word> &message, std::string label) {
  return std::make_unique<EnginePot>(candidates, message.data(),
                                     message.size() * sizeof(capnp::word),
                                     std::move(label));
}

abi::CandidateOptions engineSearch() {
  abi::CandidateOptions opts;
  opts.include_dir_envs = true;
  opts.exclusive_explicit = true;
  return opts;
}

/// Candidate paths of the generic engine library named @p stem.
std::vector<std::string> engineCandidates(const std::string &stem,
                                          const std::string &explicit_path,
                                          const std::vector<std::string> &envs) {
  std::vector<std::string> out;
  if (!explicit_path.empty()) {
    out.push_back(explicit_path);
    return out;
  }
  for (const auto &var : envs) {
    if (const char *e = std::getenv(var.c_str()); e != nullptr && *e != '\0') {
      out.emplace_back(e);
    }
  }
  for (auto &c : abi::engine_candidates(stem, "", {}, engineSearch())) {
    out.push_back(std::move(c));
  }
  return out;
}

#ifdef RGPOT_HAS_XTB
GFNMethod gfnMethod(const std::string &name) {
  const std::string m = lowered(name);
  if (m == "gfnff")
    return GFNMethod::GFNFF;
  if (m == "gfn0")
    return GFNMethod::GFN0xTB;
  if (m == "gfn1")
    return GFNMethod::GFN1xTB;
  if (m == "gfn2")
    return GFNMethod::GFN2xTB;
  throw std::invalid_argument("makePotential: xtb method '" + name +
                              "' is not one of gfnff, gfn0, gfn1, gfn2");
}
#endif

#ifdef RGPOT_HAS_EXPR
std::unique_ptr<PotentialBase> exprFrom(const ::ExprParams::Reader &params);

std::unique_ptr<PotentialBase> exprTerm(const ::PotSpec::Reader &spec,
                                        const std::string &name) {
  switch (spec.which()) {
  case ::PotSpec::EXPR:
    return exprFrom(spec.getExpr());
  case ::PotSpec::POTENTIAL:
    return rgpot::rpc::potentialFromSpec(spec.getPotential());
  default:
    throw std::invalid_argument("makePotential: expr term '" + name +
                                "' names no potential");
  }
}

std::unique_ptr<PotentialBase> exprFrom(const ::ExprParams::Reader &params) {
  std::vector<ExprPot::Term> terms;
  for (const auto term : params.getTerms()) {
    const std::string name = term.getName().cStr();
    terms.emplace_back(name, exprTerm(term.getPot(), name));
  }
  return std::make_unique<ExprPot>(params.getExpression().cStr(), std::move(terms));
}
#endif

std::unique_ptr<PotentialBase> mopacFrom(const ::MOPACParams::Reader &p) {
  // MOPACPot takes these six; refuse the fields it would drop rather than
  // run another calculation than the one asked for.
  std::vector<std::string> dropped;
  if (p.getEpsilon() != 1.0)
    dropped.emplace_back("epsilon");
  if (p.getSolver() != 0)
    dropped.emplace_back("solver");
  if (p.getExtraKeywords().size() != 0)
    dropped.emplace_back("extraKeywords");
  if (p.hasCell())
    dropped.emplace_back("cell");
  if (!dropped.empty()) {
    std::string names;
    for (const auto &d : dropped)
      names += (names.empty() ? "" : ", ") + d;
    throw std::invalid_argument("makePotential: MOPACPot does not take MOPACParams " +
                                names);
  }
  MOPACPot::Config c;
  c.charge = p.getCharge();
  c.spin = p.getSpin();
  c.model = p.getModel();
  c.tolerance = p.getTolerance();
  c.max_time = p.getMaxTime();
  c.engine_path = p.getEnginePath().cStr();
  return std::make_unique<MOPACPot>(c);
}

std::string trimmed(const std::string &s) {
  const auto first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

void appendWords(std::string &to, const std::string &words) {
  if (words.empty())
    return;
  to += (to.empty() ? "" : " ") + words;
}

/// ORCAParams lowered onto the OrcaCConfig fields of liborcac ABI 1. orcac
/// takes the method as one token, so words after the first join keywords
/// ("UKS PBE0" is method UKS, keyword PBE0, the same `!` line for ORCA).
/// Keywords on one line that does not start with '%' stay keywords, without
/// a leading '!'. Otherwise each line starting with '!' joins keywords and
/// the other lines are the verbatim blocks.
ORCACConfig orcacFrom(const ::ORCAParams::Reader &p) {
  if (p.getKeepFiles()) {
    throw std::invalid_argument("makePotential: ORCACPot does not take ORCAParams "
                                "keepFiles; liborcac ABI 1 always removes its "
                                "run directory");
  }
  ORCACConfig c;
  const std::string method = trimmed(p.getMethod().cStr());
  const auto split = method.find_first_of(" \t");
  c.method = method.substr(0, split);
  if (split != std::string::npos)
    appendWords(c.keywords, trimmed(method.substr(split)));
  c.basis = trimmed(p.getBasis().cStr());
  const std::string keywords = trimmed(p.getKeywords().cStr());
  if (keywords.find('\n') == std::string::npos && keywords.rfind('%', 0) != 0) {
    appendWords(c.keywords, trimmed(keywords.rfind('!', 0) == 0 ? keywords.substr(1)
                                                                 : keywords));
  } else {
    std::size_t start = 0;
    while (start <= keywords.size()) {
      auto end = keywords.find('\n', start);
      if (end == std::string::npos)
        end = keywords.size();
      const std::string line = keywords.substr(start, end - start);
      const std::string word = trimmed(line);
      if (word.rfind('!', 0) == 0)
        appendWords(c.keywords, trimmed(word.substr(1)));
      else if (!word.empty())
        c.blocks += line + "\n";
      start = end + 1;
    }
  }
  c.executable = p.getExecutable().cStr();
  c.scratch_dir = p.getScratchDir().cStr();
  c.charge = p.getCharge();
  c.multiplicity = p.getMultiplicity();
  c.nprocs = p.getNprocs();
  c.maxcore_mb = p.getMaxcoreMb();
  c.timeout_s = p.getTimeoutS();
  c.restart = p.getRestart();
  return c;
}

} // namespace

std::unique_ptr<PotentialBase> makePotential(const ::PotentialConfig::Reader &cfg) {
  switch (cfg.which()) {
  case ::PotentialConfig::NONE:
    throw std::invalid_argument(
        "makePotential: the PotentialConfig carries no backend arm (none)");

  case ::PotentialConfig::LJ: {
    const auto p = cfg.getLj();
    LJConfig c;
    c.u0 = p.getU0();
    c.psi = p.getPsi();
    c.cutoff = p.getCutoff();
    c.switch_width = p.getSwitchWidth();
    return std::make_unique<LJPot>(c);
  }
  case ::PotentialConfig::MORSE: {
    MorseConfig c = rgpot::rpc::morseConfigFrom(cfg.getMorse());
    c.switch_width = cfg.getMorse().getSwitchWidth();
    return std::make_unique<MorsePot>(c);
  }
  case ::PotentialConfig::ZBL: {
    ZBLConfig c;
    c.cut_inner = cfg.getZbl().getCutInner();
    c.cut_global = cfg.getZbl().getCutGlobal();
    return std::make_unique<ZBLPot>(c);
  }
  case ::PotentialConfig::D2:
    return std::make_unique<D2Pot>(rgpot::rpc::d2ConfigFrom(cfg.getD2()));
  case ::PotentialConfig::D2CPMD:
    return std::make_unique<D2CpmdPot>(rgpot::rpc::d2CpmdConfigFrom(cfg.getD2cpmd()));
  case ::PotentialConfig::MULLER_BROWN:
    return std::make_unique<MullerBrownPot>(
        rgpot::rpc::mullerBrownConfigFrom(cfg.getMullerBrown()));
  case ::PotentialConfig::COMPOSITE:
    return rgpot::rpc::compositeFromParams(cfg.getComposite());
  case ::PotentialConfig::CPMD:
    return std::make_unique<CPMDPot>(cfg.getCpmd());
  case ::PotentialConfig::NWCHEM:
    return std::make_unique<NWChemPot>(cfg.getNwchem());
  case ::PotentialConfig::MOPAC:
    return mopacFrom(cfg.getMopac());
  case ::PotentialConfig::LAMMPS: {
    // The lammpc profile reads the whole PotentialConfig with the lammps arm.
    const auto message = flatMessage(cfg);
    return rgpot_lammpc_potential(message.data(), message.size() * sizeof(capnp::word));
  }

  case ::PotentialConfig::D3: {
#ifdef RGPOT_HAS_DFTD3
    const auto p = cfg.getD3();
    D3Config c;
    c.functional = p.getFunctional().cStr();
    const std::string damping = lowered(p.getDamping().cStr());
    if (damping == "bj") {
      c.damping = D3Damping::BJ;
    } else if (damping == "zero") {
      c.damping = D3Damping::Zero;
    } else {
      throw std::invalid_argument("makePotential: d3 damping '" + damping +
                                  "' is not bj or zero");
    }
    c.atm = p.getAtm();
    return std::make_unique<D3Pot>(c);
#else
    notBuilt("d3", "-Dwith_dftd3=true");
#endif
  }
  case ::PotentialConfig::D4:
#ifdef RGPOT_HAS_DFTD4
    return std::make_unique<D4Pot>(rgpot::rpc::d4ConfigFrom(cfg.getD4()));
#else
    notBuilt("d4", "-Dwith_dftd4=true");
#endif
  case ::PotentialConfig::XTB: {
#ifdef RGPOT_HAS_XTB
    const auto p = cfg.getXtb();
    XTBConfig c;
    c.method = gfnMethod(p.getMethod().cStr());
    c.accuracy = p.getAccuracy();
    c.electronic_temperature = p.getElectronicTemperature();
    c.max_iterations = p.getMaxIterations();
    c.charge = p.getCharge();
    c.uhf = p.getUhf();
    return std::make_unique<XTBPot>(c);
#else
    // libxtb_engine reads BuiltinParams naming xtb with the xtb arm.
    capnp::MallocMessageBuilder message;
    auto builtin = message.initRoot<::BuiltinParams>();
    builtin.setName("xtb");
    builtin.initConfig().setXtb(cfg.getXtb());
    const auto words = capnp::messageToFlatArray(message);
    return enginePot(engineCandidates("xtb_engine", "", {"RGPOT_XTB_ENGINE", "XTB_ENGINE"}),
                     std::vector<capnp::word>(words.begin(), words.end()), "xtb");
#endif
  }
  case ::PotentialConfig::TBLITE: {
#ifdef RGPOT_HAS_TBLITE
    const auto p = cfg.getTblite();
    TBLiteConfig c;
    const std::string method = lowered(p.getMethod().cStr());
    if (method == "gfn1") {
      c.method = TBLiteMethod::GFN1;
    } else if (method == "gfn2") {
      c.method = TBLiteMethod::GFN2;
    } else if (method == "ipea1") {
      c.method = TBLiteMethod::IPEA1;
    } else {
      throw std::invalid_argument("makePotential: tblite method '" + method +
                                  "' is not one of gfn1, gfn2, ipea1");
    }
    c.accuracy = p.getAccuracy();
    c.electronic_temperature = p.getElectronicTemperature();
    c.max_iterations = p.getMaxIterations();
    c.charge = p.getCharge();
    c.uhf = p.getUhf();
    return std::make_unique<TBLitePot>(c);
#else
    notBuilt("tblite", "-Dwith_tblite=true");
#endif
  }
  case ::PotentialConfig::SKALA: {
#ifdef RGPOT_HAS_SKALA
    const auto p = cfg.getSkala();
    SkalaConfig c;
    c.xc = p.getXc().cStr();
    c.basis = p.getBasis().cStr();
    c.charge = p.getCharge();
    c.multiplicity = p.getMultiplicity();
    c.engine_path = p.getEnginePath().cStr();
    c.nwchem_root = p.getNwchemRoot().cStr();
    return std::make_unique<SkalaPot>(c);
#else
    notBuilt("skala", "-Dwith_skala=true");
#endif
  }
  case ::PotentialConfig::CP2K:
#ifdef RGPOT_HAS_CP2K
    return std::make_unique<CP2KPot>(cfg.getCp2k());
#else
    notBuilt("cp2k", "-Dwith_cp2k=true");
#endif
  case ::PotentialConfig::EXPR:
#ifdef RGPOT_HAS_EXPR
    return exprFrom(cfg.getExpr());
#else
    notBuilt("expr", "-Dwith_expr=true");
#endif

  case ::PotentialConfig::METATOMIC: {
    const auto p = cfg.getMetatomic();
    return enginePot(engineCandidates("metatomic_engine", p.getEnginePath().cStr(),
                                      {"RGPOT_METATOMIC_ENGINE", "METATOMIC_ENGINE"}),
                     flatMessage(p), "metatomic");
  }
  case ::PotentialConfig::UMA:
    return enginePot(engineCandidates("uma_engine", "", {"RGPOT_UMA_ENGINE"}),
                     flatMessage(cfg.getUma()), "uma");
  case ::PotentialConfig::ORCA:
    return std::make_unique<ORCACPot>(orcacFrom(cfg.getOrca()));
  case ::PotentialConfig::PLUGIN: {
    const auto p = cfg.getPlugin();
    const std::string name = p.getPluginName().cStr();
    const std::string library = p.getLibraryPath().cStr();
    if (name.empty() && library.empty()) {
      throw std::invalid_argument(
          "makePotential: the plugin arm names neither pluginName nor libraryPath");
    }
    const auto payload = p.getOpaquePayload();
    // Engines read the payload as capnp words, so it is copied to a
    // word-aligned buffer.
    std::vector<capnp::word> message((payload.size() + sizeof(capnp::word) - 1) /
                                     sizeof(capnp::word));
    if (payload.size() != 0) {
      std::memcpy(message.data(), payload.begin(), payload.size());
    }
    auto candidates = engineCandidates(name + "_engine", library,
                                       {"RGPOT_" + upper(name) + "_ENGINE"});
    auto pot = std::make_unique<EnginePot>(
        candidates, message.data(), payload.size(),
        "plugin " + (name.empty() ? library : name));
    return pot;
  }

  case ::PotentialConfig::GPR:
    throw std::invalid_argument(
        "makePotential: the gpr arm configures a surrogate its host (gpridge) "
        "builds; rgpot has no gpr backend");
  case ::PotentialConfig::POTLIB:
    throw std::invalid_argument(
        "makePotential: the potlib arm has no rgpot backend; libpotlibc_engine "
        "takes an RgpotPotlibcConfig, which the plugin arm can carry as its "
        "opaquePayload");
  case ::PotentialConfig::EXTERNAL:
    throw std::invalid_argument(
        "makePotential: the external arm runs an eOn-style external program, "
        "which rgpot does not host");
  }
  throw std::invalid_argument("makePotential: the " +
                              rgpot::rpc::potentialConfigArm(cfg) +
                              " arm is newer than this build's schema");
}

std::unique_ptr<PotentialBase> makePotential(const void *bytes, std::size_t size) {
  if (bytes == nullptr || size == 0 || size % sizeof(capnp::word) != 0) {
    throw std::invalid_argument(
        "makePotential: PotentialConfig bytes must be a non-empty flat array "
        "of 8-byte words");
  }
  // FlatArrayMessageReader needs word alignment; the caller's buffer may not
  // have it.
  std::vector<capnp::word> words(size / sizeof(capnp::word));
  std::memcpy(words.data(), bytes, size);
  try {
    capnp::FlatArrayMessageReader reader(kj::arrayPtr(words.data(), words.size()));
    return makePotential(reader.getRoot<::PotentialConfig>());
  } catch (const kj::Exception &e) {
    throw std::invalid_argument(std::string("makePotential: not a PotentialConfig "
                                            "message: ") +
                                e.getDescription().cStr());
  }
}

} // namespace rgpot
