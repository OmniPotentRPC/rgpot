// MIT License — SkalaPot: NWChem DFT, XC keyword from SkalaConfig

#include "rgpot/SkalaPot/SkalaPot.hpp"

#include <capnp/message.h>

#include <stdexcept>

namespace rgpot {

namespace {

void fill_skala_params(::NWChemParams::Builder p, const SkalaConfig &cfg) {
  p.setTheory("dft");
  p.setBasis(cfg.basis);
  p.setCharge(cfg.charge);
  p.setMultiplicity(cfg.multiplicity);
  p.setTask("gradient");
  if (!cfg.engine_path.empty())
    p.setEnginePath(cfg.engine_path);
  if (!cfg.nwchem_root.empty())
    p.setNwchemRoot(cfg.nwchem_root);
  auto stanzas = p.initInputStanzas(1);
  auto stanza = stanzas[0];
  stanza.setKind(::NWChemInputStanza::Kind::DFT);
  auto dft = stanza.initDft();
  dft.setXc(cfg.xc);
}

} // namespace

void SkalaPot::recomputeParamsKey() {
  Fnv1a fp;
  fp.u64(kKernelVersion);
  fp.str(m_config.xc);
  fp.str(m_config.basis);
  fp.i64(m_config.charge);
  fp.i64(m_config.multiplicity);
  fp.str(m_config.engine_path);
  fp.str(m_config.nwchem_root);
  m_paramsKey = fp.h;
}

void SkalaPot::applyToEngine() {
  if (m_config.basis.empty())
    throw std::runtime_error("SkalaPot: basis is required");
  if (m_config.xc.empty())
    throw std::runtime_error("SkalaPot: xc is required");
  if (m_config.multiplicity < 1)
    throw std::runtime_error("SkalaPot: multiplicity must be >= 1");
  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::NWChemParams>();
  fill_skala_params(p, m_config);
  (void)m_nwchem.setParams(p.asReader());
  recomputeParamsKey();
}

SkalaPot::SkalaPot() : SkalaPot(SkalaConfig{}) {}

SkalaPot::SkalaPot(const SkalaConfig &config)
    : Potential(PotType::Skala), m_config(config), m_nwchem() {
  applyToEngine();
}

void SkalaPot::setChargeMultiplicity(int charge, int multiplicity) {
  m_config.charge = charge;
  m_config.multiplicity = multiplicity;
  applyToEngine();
}

void SkalaPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  m_nwchem.forceImpl(in, out);
}

} // namespace rgpot
