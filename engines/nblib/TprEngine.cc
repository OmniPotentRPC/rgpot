// MIT License
// Copyright 2026 Rohit Goswami

#include "TprEngine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

#include "gromacs/domdec/dlbtiming.h"
#include "gromacs/domdec/domdec.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/fileio/tpxio.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/hardware/cpuinfo.h"
#include "gromacs/hardware/hardwaretopology.h"
#include "gromacs/hardware/hw_info.h"
#include "gromacs/listed_forces/listed_forces.h"
#include "gromacs/math/arrayrefwithpadding.h"
#include "gromacs/math/paddedvector.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/dispersioncorrection.h"
#include "gromacs/mdlib/enerdata_utils.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/forcerec.h"
#include "gromacs/mdlib/freeenergyparameters.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdlib/vsite.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/listoflists.h"
#include "gromacs/utility/logger.h"

namespace rgpot::nblibpot {

namespace {

// Distance an atom may travel before the pair list is rebuilt is half of this.
constexpr double kSkin = 0.1; // nm

void refuse(bool condition, const char *what) {
  if (condition) {
    throw std::invalid_argument(std::string("the TPR uses ") + what +
                                ", which the TPR session does not evaluate");
  }
}

void checkSupported(const t_inputrec &ir, const gmx_mtop_t &mtop) {
  refuse(ir.efep != FreeEnergyPerturbationType::No, "free-energy perturbation");
  refuse(ir.bPull, "pulling");
  refuse(ir.bDoAwh, "AWH");
  refuse(ir.bRot, "enforced rotation");
  refuse(ir.bIMD, "interactive MD");
  refuse(ir.bQMMM, "QM/MM");
  refuse(ir.useMts, "multiple time stepping");
  refuse(ir.nwall != 0, "walls");
  refuse(ir.bSimTemp, "simulated tempering");
  refuse(ir.eSwapCoords != SwapType::No, "coordinate swapping");
  refuse(ir.cos_accel != 0, "cosine acceleration");
  refuse(ir_haveBoxDeformation(ir), "box deformation");
  refuse(ir.pbcType != PbcType::Xyz, "a cell that is not periodic in x, y and z");
  refuse(ir.epsilon_surface != 0 || ir.ewald_geometry != EwaldGeometry::ThreeD,
         "an Ewald surface correction");
  refuse(usingLJPme(ir.vdwtype), "LJ-PME");
  refuse(ir.coulombtype != CoulombInteractionType::Pme &&
             ir.coulombtype != CoulombInteractionType::Cut &&
             !usingRF(ir.coulombtype),
         "electrostatics other than PME, cutoff or reaction field");
  refuse(ir.pressureCouplingOptions.epc != PressureCoupling::No,
         "pressure coupling");
  for (const auto &freeze : {ir.opts.nFreeze}) {
    for (int g = 0; freeze != nullptr && g < ir.opts.ngfrz; ++g) {
      refuse(freeze[g][XX] != 0 || freeze[g][YY] != 0 || freeze[g][ZZ] != 0,
             "freeze groups");
    }
  }
  for (const int ftype : {F_DISRES, F_ORIRES, F_DIHRES, F_COM_PULL}) {
    refuse(gmx_mtop_ftype_count(mtop, ftype) != 0, "NMR-type restraints");
  }
  for (const int ftype : {F_TABBONDS, F_TABBONDSNC, F_TABANGLES, F_TABDIHS}) {
    refuse(gmx_mtop_ftype_count(mtop, ftype) != 0, "tabulated listed terms");
  }
  refuse(gmx_mtop_ftype_count(mtop, F_THOLE_POL) != 0 ||
             gmx_mtop_ftype_count(mtop, F_POLARIZATION) != 0 ||
             gmx_mtop_ftype_count(mtop, F_WATER_POL) != 0,
         "polarizable sites");
}

void setThreads(int threads) {
  for (int m = 0; m < static_cast<int>(ModuleMultiThread::Count); ++m) {
    gmx_omp_nthreads_set(static_cast<ModuleMultiThread>(m), threads);
  }
}

} // namespace

class TprEngine::Impl {
public:
  Impl(const std::string &path, int threads)
      : hardware_(std::make_unique<gmx::CpuInfo>(gmx::CpuInfo::detect()),
                  std::make_unique<gmx::HardwareTopology>(
                      gmx::HardwareTopology::detect())) {
    if (threads < 1) {
      throw std::invalid_argument("the TPR session needs at least one thread");
    }
    if (!std::filesystem::exists(path)) {
      throw std::invalid_argument("there is no TPR at " + path);
    }
    setThreads(threads);
    read_tpx_state(path, &ir_, &state_, &mtop_);
    checkSupported(ir_, mtop_);

    // No dynamic pruning and an explicit buffer: the list is rebuilt from
    // the displacements, so no estimate of the diffusion enters.
    ir_.verletbuf_tol = -1;
    ir_.rlist = std::max(ir_.rcoulomb, ir_.rvdw) + kSkin;
    copy_mat(state_.box, box_);
    requireCutoff();

    commrec_.duty = DUTY_PP | DUTY_PME;
    forcerec_.forceProviders = &providers_;
    // mdrun passes this name even when the file is never opened.
    // Dispersion correction aborts on a null name. A cutoff or a
    // modifier builds the table in memory; only a user table reads
    // the file.
    init_forcerec(nullptr, logger_, simulationWork_, &forcerec_, ir_, mtop_,
                  &commrec_, box_, "table.xvg", nullptr,
                  gmx::ArrayRef<const std::string>{}, -1);
    disres_ = std::make_unique<t_disresdata>();
    disres_->nres = 0;
    forcerec_.fcdata->disres = disres_.get();

    top_ = std::make_unique<gmx_localtop_t>(mtop_.ffparams);
    gmx_mtop_generate_local_top(mtop_, top_.get(), false);
    total_ = mtop_.natoms;
    mdAtoms_ = gmx::makeMDAtoms(nullptr, mtop_, ir_, false);
    atoms2md(mtop_, ir_, -1, {}, total_, mdAtoms_.get());
    update_mdatoms(mdAtoms_->mdatoms(), 0.0);

    x_.resizeWithPadding(total_);
    f_.resizeWithPadding(total_);
    std::copy(state_.x.begin(), state_.x.begin() + total_, x_.begin());

    vsites_ = gmx::makeVirtualSitesHandler(mtop_, &commrec_, ir_.pbcType, {});
    if (vsites_ != nullptr) {
      vsites_->setVirtualSites(top_->idef.il, total_,
                               mdAtoms_->mdatoms()->homenr,
                               mdAtoms_->mdatoms()->ptype);
    }
    forcerec_.nbv = gmx::init_nb_verlet(
        logger_, ir_, forcerec_, &commrec_, hardware_, false, nullptr, mtop_,
        false, nullptr, gmx::makeConstArrayRef(state_.x), box_, nullptr);
    forcerec_.listedForces[0].setup(top_->idef, total_, false, {}, 0);
    for (auto &buffers : forcerec_.forceHelperBuffers) {
      buffers.resize(total_);
    }
    forcerec_.longRangeNonbondeds = std::make_unique<CpuPpLongRangeNonbondeds>(
        forcerec_.n_tpi, forcerec_.ic->ewaldcoeff_q, forcerec_.ic->epsilon_r,
        forcerec_.qsum, forcerec_.ic->eeltype, forcerec_.ic->vdwtype, ir_,
        &nrnb_, nullptr, nullptr);
    forcerec_.longRangeNonbondeds->updateAfterPartition(*mdAtoms_->mdatoms());
    if (usingPme(ir_.coulombtype)) {
      initPme(threads);
    }
    enerd_ = std::make_unique<gmx_enerdata_t>(ir_.opts.ngener, nullptr);
    setupWork();
    mapAtoms();
  }

  ~Impl() {
    if (forcerec_.pmedata != nullptr) {
      gmx_pme_destroy(forcerec_.pmedata);
      forcerec_.pmedata = nullptr;
    }
    forcerec_.fcdata->disres = nullptr;
  }

  double evaluate(const double *positions, double *forces) {
    for (std::size_t k = 0; k < massive_.size(); ++k) {
      for (int d = 0; d < DIM; ++d) {
        x_[massive_[k]][d] = static_cast<real>(positions[3 * k + d]);
      }
    }
    if (vsites_ != nullptr) {
      vsites_->construct(x_.arrayRefWithPadding().unpaddedArrayRef(), {}, box_,
                         gmx::VSiteOperation::Positions);
    }
    auto all = x_.arrayRefWithPadding().unpaddedArrayRef();
    if (listValid_ && moved()) {
      listValid_ = false;
    }
    if (!listValid_) {
      rebuild();
    }
    // Between searches an atom keeps the image it had at the search, which
    // the pair-list shifts assume.
    for (int i = 0; i < total_; ++i) {
      for (int d = 0; d < DIM; ++d) {
        x_[i][d] += imageShift_[i][d];
      }
    }
    auto *nbv = forcerec_.nbv.get();
    nbv->convertCoordinates(gmx::AtomLocality::Local, all);

    reset_enerdata(enerd_.get());
    auto forceRef = f_.arrayRefWithPadding();
    std::fill(f_.begin(), f_.end(), gmx::RVec(0, 0, 0));
    gmx::ForceWithShiftForces shifted(forceRef, false, {});
    gmx::ForceWithVirial direct(forceRef.unpaddedArrayRef(), false);
    gmx::ForceOutputs outputs(shifted, false, direct);

    nbv->dispatchNonbondedKernel(
        gmx::InteractionLocality::Local, *forcerec_.ic, step_,
        gmx::enbvClearFYes, forcerec_.shift_vec,
        enerd_->grpp.energyGroupPairTerms[NonBondedEnergyTerms::LJSR],
        enerd_->grpp.energyGroupPairTerms[NonBondedEnergyTerms::CoulombSR],
        &nrnb_);
    nbv->atomdata_add_nbat_f_to_f(gmx::AtomLocality::All,
                                  forceRef.unpaddedArrayRef());

    t_pbc pbc;
    set_pbc_dd(&pbc, ir_.pbcType, nullptr, TRUE, box_);
    const auto *md = mdAtoms_->mdatoms();
    forcerec_.listedForces[0].calculate(
        nullptr, box_, &commrec_, nullptr, x_.constArrayRefWithPadding(), {},
        forcerec_.fcdata.get(), nullptr, &outputs, &forcerec_, &pbc,
        enerd_.get(), &nrnb_, lambda_, md->chargeA, md->chargeB,
        gmx::makeConstArrayRef(md->bPerturbed), md->cENER, md->nPerturbed,
        nullptr, step_);

    const std::array<gmx::RVec, 2> dipole{gmx::RVec(0, 0, 0),
                                          gmx::RVec(0, 0, 0)};
    const DDBalanceRegionHandler balance(nullptr);
    forcerec_.longRangeNonbondeds->calculate(
        forcerec_.pmedata, &commrec_, x_.constArrayRefWithPadding().unpaddedConstArrayRef(),
        &outputs.forceWithVirial(), enerd_.get(), box_, lambda_, dipole, step_,
        balance);
    if (vsites_ != nullptr) {
      matrix unused = {{0}};
      vsites_->spreadForces(
          x_.constArrayRefWithPadding().unpaddedConstArrayRef(),
          forceRef.unpaddedArrayRef(),
          gmx::VirtualSitesHandler::VirialHandling::None, {}, unused, &nrnb_,
          box_, nullptr);
    }
    // The correction is a function of the density. It has no per-atom
    // force. calculate() also returns a scalar virial and a pressure.
    // mdrun adds that virial to the diagonal of the virial tensor and
    // stores the pressure in F_PDISPCORR, which sits outside the
    // potential sum. This session returns energy and forces only, so
    // the virial is not applied and the pressure is not returned.
    // Ener leaves both at zero. EnerPres fills both, and they are
    // still unused here. F_DISPCORR is inside the sum, so the
    // potential below includes it.
    if (forcerec_.dispersionCorrection) {
      const DispersionCorrection::Correction correction =
          forcerec_.dispersionCorrection->calculate(
              box_, lambda_[static_cast<int>(
                        FreeEnergyPerturbationCouplingType::Vdw)]);
      enerd_->term[F_DISPCORR] = correction.energy;
      enerd_->term[F_PDISPCORR] = correction.pressure;
    }
    accumulatePotentialEnergies(enerd_.get(), lambda_, ir_.fepvals.get());
    terms_.ljSr = enerd_->term[F_LJ];
    terms_.coulombSr = enerd_->term[F_COUL_SR];
    terms_.coulombReciprocal = enerd_->term[F_COUL_RECIP];
    terms_.dispersionCorrection = enerd_->term[F_DISPCORR];
    terms_.potential = enerd_->term[F_EPOT];

    for (std::size_t k = 0; k < massive_.size(); ++k) {
      for (int d = 0; d < DIM; ++d) {
        forces[3 * k + d] = f_[massive_[k]][d];
      }
    }
    const double energy = enerd_->term[F_EPOT];
    if (!std::isfinite(energy)) {
      throw std::runtime_error("the TPR session met a non-finite energy");
    }
    return energy;
  }

  std::array<double, 9> box() const {
    std::array<double, 9> out{};
    for (int i = 0; i < DIM; ++i) {
      for (int j = 0; j < DIM; ++j) {
        out[3 * i + j] = box_[i][j];
      }
    }
    return out;
  }

  void setBox(const std::array<double, 9> &box) {
    if (box[1] != 0.0 || box[2] != 0.0 || box[5] != 0.0) {
      throw std::invalid_argument(
          "a GROMACS box is lower triangular: a along x, b in the xy plane");
    }
    const std::array<double, 9> now = this->box();
    // The PME grid and its spline tables were chosen for the TPR's cell, so
    // a different cell is another model rather than another state.
    for (std::size_t k = 0; k < 9; ++k) {
      if (std::abs(box[k] - now[k]) > 1.0e-6) {
        throw std::invalid_argument(
            "the TPR session keeps the cell of its TPR; write a new TPR for "
            "another cell");
      }
    }
  }

  std::vector<int> massive_;
  std::vector<double> masses_;
  std::vector<double> initialX_;
  std::vector<double> initialV_;
  std::vector<md::DistanceConstraint> constraints_;
  Terms terms() const { return terms_; }

private:
  void requireCutoff() const {
    if (!(ir_.rlist * ir_.rlist < max_cutoff2(ir_.pbcType, box_))) {
      throw std::invalid_argument(
          "the TPR cell is too small for its cutoff and the pair-list buffer");
    }
  }

  void initPme(int threads) {
    const NumPmeDomains domains = getNumPmeDomains(nullptr);
    forcerec_.pmedata = gmx_pme_init(
        &commrec_, domains, &ir_, box_, 0, false, false, true,
        forcerec_.ic->ewaldcoeff_q, forcerec_.ic->ewaldcoeff_lj, threads,
        PmeRunMode::CPU, nullptr, nullptr, nullptr, nullptr, logger_, nullptr);
    // Without this PME spreads no atoms and returns zero without an error.
    const auto *md = mdAtoms_->mdatoms();
    gmx_pme_reinit_atoms(forcerec_.pmedata, md->homenr, md->chargeA,
                         md->chargeB);
  }

  void setupWork() {
    simulationWork_.computeNonbonded = true;
    step_.stateChanged = true;
    step_.computeSlowForces = true;
    step_.computeEnergy = true;
    step_.computeForces = true;
    step_.computeNonbondedForces = true;
    step_.computeListedForces = true;
    std::fill(lambda_.begin(), lambda_.end(), real(0));
  }

  // Whether any atom left the region the pair list covers. Positions are
  // compared in the images the caller gave, which are continuous.
  bool moved() const {
    const double limit2 = 0.25 * kSkin * kSkin;
    double worst = 0.0;
    for (int i = 0; i < total_; ++i) {
      double d2 = 0.0;
      for (int d = 0; d < DIM; ++d) {
        const double delta = x_[i][d] - searched_[i][d];
        d2 += delta * delta;
      }
      worst = std::max(worst, d2);
    }
    return worst > limit2;
  }

  void rebuild() {
    auto all = x_.arrayRefWithPadding().unpaddedArrayRef();
    searched_.assign(all.begin(), all.end());
    put_atoms_in_box_omp(ir_.pbcType, box_, false, nullptr, all, {}, 1);
    imageShift_.resize(total_);
    for (int i = 0; i < total_; ++i) {
      for (int d = 0; d < DIM; ++d) {
        imageShift_[i][d] = x_[i][d] - searched_[i][d];
      }
    }
    auto *nbv = forcerec_.nbv.get();
    calc_shifts(box_, forcerec_.shift_vec);
    gmx::nbnxn_atomdata_copy_shiftvec(true, forcerec_.shift_vec, &nbv->nbat());
    const gmx::RVec lower(0, 0, 0);
    const gmx::RVec upper(box_[XX][XX], box_[YY][YY], box_[ZZ][ZZ]);
    nbv->putAtomsOnGrid(box_, 0, lower, upper, nullptr, {0, total_}, total_, -1,
                        forcerec_.atomInfo, all, nullptr);
    const auto *md = mdAtoms_->mdatoms();
    nbv->setAtomProperties(md->typeA, md->chargeA, forcerec_.atomInfo);
    nbv->constructPairlist(gmx::InteractionLocality::Local, top_->excls, 0,
                           &nrnb_);
    // Hand the caller's images back; evaluate applies the shifts.
    std::copy(searched_.begin(), searched_.end(), x_.begin());
    listValid_ = true;
  }

  // Bring b to the image nearest a, in a lower-triangular cell.
  void nearest(const double *a, double *b) const {
    for (int k = DIM - 1; k >= 0; --k) {
      const double n = std::round((b[k] - a[k]) / box_[k][k]);
      for (int d = 0; d <= k; ++d) {
        b[d] -= n * box_[k][d];
      }
    }
  }

  void mapAtoms() {
    const auto *md = mdAtoms_->mdatoms();
    std::vector<int> index(total_, -1);
    for (int i = 0; i < total_; ++i) {
      if (md->ptype[i] == ParticleType::VSite) {
        continue;
      }
      refuse(md->ptype[i] != ParticleType::Atom, "shell particles");
      refuse(!(md->massT[i] > 0), "atoms without mass that are no virtual sites");
      index[i] = static_cast<int>(massive_.size());
      massive_.push_back(i);
      masses_.push_back(md->massT[i]);
      for (int d = 0; d < DIM; ++d) {
        initialX_.push_back(state_.x[i][d]);
        initialV_.push_back(state_.v.empty() ? 0.0 : state_.v[i][d]);
      }
    }
    const auto add = [&](int a, int b, double length) {
      refuse(!(length > 0.0), "flexible constraints");
      if (index[a] < 0 || index[b] < 0) {
        throw std::invalid_argument("a constraint acts on a virtual site");
      }
      constraints_.push_back({static_cast<std::size_t>(index[a]),
                              static_cast<std::size_t>(index[b]), length});
    };
    const auto &idef = top_->idef;
    for (const int ftype : {F_CONSTR, F_CONSTRNC}) {
      const auto &atoms = idef.il[ftype].iatoms;
      for (std::size_t k = 0; k + 2 < atoms.size(); k += 3) {
        add(atoms[k + 1], atoms[k + 2], idef.iparams[atoms[k]].constr.dA);
      }
    }
    const auto &settles = idef.il[F_SETTLE].iatoms;
    for (std::size_t k = 0; k + 3 < settles.size(); k += 4) {
      const auto &p = idef.iparams[settles[k]].settle;
      add(settles[k + 1], settles[k + 2], p.doh);
      add(settles[k + 1], settles[k + 3], p.doh);
      add(settles[k + 2], settles[k + 3], p.dhh);
    }
    gather();
  }

  // One periodic image per constrained group: walk each group from its first
  // atom and place every neighbour nearest the atom that reached it.
  void gather() {
    std::vector<std::vector<std::size_t>> joined(massive_.size());
    for (const auto &c : constraints_) {
      joined[c.first].push_back(c.second);
      joined[c.second].push_back(c.first);
    }
    std::vector<bool> placed(massive_.size(), false);
    std::vector<std::size_t> stack;
    for (std::size_t root = 0; root < massive_.size(); ++root) {
      if (placed[root]) {
        continue;
      }
      placed[root] = true;
      stack.assign(1, root);
      while (!stack.empty()) {
        const std::size_t at = stack.back();
        stack.pop_back();
        for (const std::size_t next : joined[at]) {
          if (!placed[next]) {
            nearest(&initialX_[3 * at], &initialX_[3 * next]);
            placed[next] = true;
            stack.push_back(next);
          }
        }
      }
    }
    // A cycle that closes only through the cell keeps a wrong length here.
    for (const auto &c : constraints_) {
      double r2 = 0.0;
      for (int d = 0; d < 3; ++d) {
        const double delta =
            initialX_[3 * c.first + d] - initialX_[3 * c.second + d];
        r2 += delta * delta;
      }
      if (std::abs(std::sqrt(r2) - c.length) > 0.1 * c.length) {
        throw std::invalid_argument(
            "a constrained group spans the periodic cell, or the TPR "
            "coordinates do not satisfy its constraints");
      }
    }
  }

  gmx_hw_info_t hardware_;
  gmx::MDLogger logger_;
  t_inputrec ir_;
  t_state state_;
  gmx_mtop_t mtop_;
  matrix box_;
  t_commrec commrec_;
  gmx::SimulationWorkload simulationWork_;
  gmx::StepWorkload step_;
  gmx::ForceProviders providers_;
  t_forcerec forcerec_;
  std::unique_ptr<t_disresdata> disres_;
  std::unique_ptr<gmx_localtop_t> top_;
  std::unique_ptr<gmx::MDAtoms> mdAtoms_;
  std::unique_ptr<gmx::VirtualSitesHandler> vsites_;
  std::unique_ptr<gmx_enerdata_t> enerd_;
  t_nrnb nrnb_;
  std::array<real, static_cast<int>(FreeEnergyPerturbationCouplingType::Count)>
      lambda_{};
  gmx::PaddedVector<gmx::RVec> x_;
  gmx::PaddedVector<gmx::RVec> f_;
  std::vector<gmx::RVec> searched_;
  std::vector<gmx::RVec> imageShift_;
  bool listValid_ = false;
  int total_ = 0;
  Terms terms_{};
};

TprEngine::TprEngine(const std::string &path, int threads)
    : impl_(std::make_unique<Impl>(path, threads)) {}
TprEngine::~TprEngine() = default;
std::size_t TprEngine::atoms() const { return impl_->massive_.size(); }
const std::vector<double> &TprEngine::masses() const { return impl_->masses_; }
const std::vector<md::DistanceConstraint> &TprEngine::constraints() const {
  return impl_->constraints_;
}
const std::vector<double> &TprEngine::initialPositions() const {
  return impl_->initialX_;
}
const std::vector<double> &TprEngine::initialVelocities() const {
  return impl_->initialV_;
}
std::array<double, 9> TprEngine::box() const { return impl_->box(); }
void TprEngine::setBox(const std::array<double, 9> &box) { impl_->setBox(box); }
double TprEngine::evaluate(const double *positions, double *forces) {
  return impl_->evaluate(positions, forces);
}
TprEngine::Terms TprEngine::terms() const { return impl_->terms(); }

} // namespace rgpot::nblibpot
