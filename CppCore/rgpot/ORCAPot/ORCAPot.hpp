#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file ORCAPot.hpp
 * @brief ORCA single-point energy and gradient behind the Potential interface.
 *
 * Each force call writes an ORCA input with `! EnGrad`, runs the `orca`
 * executable as a child process in its own process group and working
 * directory, and reads the `.engrad` file ORCA writes.
 * ORCAConfig::timeout_s bounds that wait (one hour by default; zero waits
 * without a limit). Past the limit the process group is sent SIGTERM and,
 * if any process in the group is still alive, SIGKILL even if the leader has
 * exited. ORCA is neither linked nor loaded. The only
 * dependency is an executable, named by ORCAConfig::executable or found as
 * `orca` on PATH.
 * The energy is converted from Hartree to eV and the gradient from
 * Hartree/Bohr to forces in eV/Angstrom with the constants of units.hpp.
 *
 * The input carries `%scf ConvForced true end` (ORCA 6.1 manual, section
 * 2.6.1, https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/scf.html),
 * so ORCA stops after an SCF that did not converge. A call whose output
 * reports such an SCF throws ORCAError even when ORCA wrote a gradient.
 *
 * Each call starts from ORCA's own guess. With ORCAConfig::restart on, the
 * converged orbitals are carried forward instead: the .gbw of one call is
 * copied to a second name and handed to the next through `%moinp` and
 * `! MORead`. Without a .gbw, or after a failure or a change of
 * composition, the next call still starts from ORCA's own guess. With
 * restart on, the result cache is not used (see setCache).
 */

#include "rgpot/Potential.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot {

/// A failure of the ORCA run itself: a stale .engrad or .gbw could not be
/// removed, the input could not be written, the executable changed since
/// the instance was made or while it ran, the process could not be started,
/// ran past ORCAConfig::timeout_s, ended with a non-zero status or a signal,
/// reported an SCF that did not converge, left no .engrad file, or wrote one
/// that is malformed or holds a non-finite number. The message carries the
/// tail of ORCA's stderr and of its output file. Input a caller got wrong (a
/// box in which the atoms would wrap, an unknown atomic number, an
/// impossible spin, a timeout that is negative or not finite) throws
/// std::invalid_argument instead.
class ORCAError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief Parameters for ORCAPot.
 *
 * Every field that shapes the calculation is the caller's: there is no
 * default method or basis, and the ORCA installation is not assumed.
 */
struct ORCAConfig {
  /// Default of timeout_s: one hour, in seconds.
  static constexpr double kDefaultTimeout = 3600.0;

  /// The orca executable: a path, or a bare name looked up on PATH. Empty
  /// looks up `orca` on PATH.
  std::string executable;
  /// ORCA method keyword (a functional, `HF`, `DLPNO-CCSD(T)`, a composite
  /// such as `r2SCAN-3c`). Required.
  std::string method;
  /// Basis set keyword. Empty for a composite method that carries its own.
  std::string basis;
  /// Further simple-input keywords, separated by blanks, for example
  /// `TightSCF D4 RIJCOSX`.
  std::string keywords;
  int charge = 0;
  /// Spin multiplicity 2S + 1.
  int multiplicity = 1;
  /// Processes for `%pal nprocs`. Above 1 ORCA starts its own MPI ranks.
  int nprocs = 1;
  /// `%maxcore`, memory per process in MB; 1024 is rgpot's choice. Zero
  /// leaves `%maxcore` out of the input, so ORCA's own default applies:
  /// 4096 MB in ORCA 6.1 (ORCA 6.1 manual, section 2.1.3,
  /// https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/input.html).
  /// A negative value is rejected.
  int maxcore_mb = 1024;
  /// Start each call from the orbitals of the previous one, through
  /// `! MORead`. Off by default, so every call starts from ORCA's own guess
  /// and its result does not depend on the calls before it. With restart
  /// on, ORCAPot does not use a result cache it is given.
  bool restart = false;
  /// Directory under which the per-instance working directory is created.
  /// Empty uses the system temporary directory.
  std::string scratch_dir;
  /// Leave the working directory in place when the instance is destroyed.
  bool keep_files = false;
  /// Seconds to wait for one orca process. The default is one hour. Zero
  /// waits until the process ends. Past the limit the process group is sent
  /// SIGTERM and, if any process in the group is still alive two seconds
  /// later, SIGKILL even if the leader has exited; the child is reaped and
  /// the call throws ORCAError. A negative or non-finite value is rejected.
  /// Not part of the result-cache key: it does not change the energy.
  double timeout_s = kDefaultTimeout;
};

/**
 * @class ORCAPot
 * @brief ORCA energy and gradient through a child process.
 * @ingroup rgpot_potentials
 */
class ORCAPot : public Potential<ORCAPot> {
public:
  /// Resolves the executable and creates the working directory.
  /// Throws std::invalid_argument for a config that cannot run (empty
  /// method, an nprocs or multiplicity below 1, a negative maxcore_mb, a
  /// line break in a keyword field, a timeout_s
  /// that is negative or not finite) and ORCAError when no executable is
  /// found or the directory cannot be made.
  explicit ORCAPot(const ORCAConfig &config);
  ~ORCAPot() override;

  ORCAPot(const ORCAPot &) = delete;
  ORCAPot &operator=(const ORCAPot &) = delete;

  /// Positions in Angstrom, energy in eV, forces in eV/Angstrom. ORCA has
  /// no periodic boundary conditions, so a box is only padding around the
  /// cluster and ORCA sees the atoms where they are. The box may be absent,
  /// all zero, or a cell in which every pair of atoms is its own nearest
  /// periodic image: along every nonzero lattice vector, not only the box
  /// vectors, the atoms span at most half its length. The cube of at least
  /// 100 Angstrom that rgpot.ase pads a cluster into is such a cell. A box
  /// with a translated image closer than the atom itself, a singular box,
  /// a cell too skewed to check (including one whose image bound is not
  /// finite or out of range), a non-finite box entry and, with a box, a
  /// non-finite position throw std::invalid_argument. The resolved
  /// executable runs, and a call throws ORCAError once it has changed (see
  /// resolvedExecutablePath()). A run that fails, including one that
  /// passes timeout_s, throws ORCAError.
  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// One instance owns one working directory and one wavefunction, so
  /// concurrent callers need one instance each. No stress, no periodicity.
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance,
            .periodic = false,
            .stress = false};
  }

  /// Hash of method, basis, keywords, charge, multiplicity, restart, and
  /// the executable path with symlinks resolved and the size, modification
  /// time, change time, device, inode and content hash of that file, all
  /// read by the constructor. Each call resolves and reads the executable
  /// again; once it no longer matches, as when a symlink is retargeted or
  /// the file is replaced or rewritten in place, the key matches no entry
  /// an unchanged instance writes. A result-cache entry made by another ORCA
  /// installation, or by the same one after it was replaced, is never
  /// served.
  [[nodiscard]] uint64_t paramsKey() const noexcept override;

#ifdef RGPOT_HAS_CACHE
  /// With ORCAConfig::restart on, the result cache is not used: a call it
  /// served would run no ORCA and leave the orbitals of an earlier call as
  /// the guess of the next one, so that call would depend on which calls
  /// happened to be cached. Without restart the cache is used as given.
  void setCache(rgpot::cache::PotentialCache *c) override {
    Potential<ORCAPot>::setCache(m_config.restart ? nullptr : c);
  }
#endif

  /// Forget the stored orbitals. The next call starts from ORCA's guess.
  void reset();

  [[nodiscard]] const ORCAConfig &config() const noexcept { return m_config; }

  /// Absolute path of the executable as named, or as found on PATH.
  [[nodiscard]] const std::string &executablePath() const noexcept {
    return m_executable;
  }

  /// The file the instance runs: executablePath() with symlinks resolved
  /// by the constructor. A call throws ORCAError once executablePath()
  /// resolves to another file or the size, modification time, change time,
  /// device, inode or content hash of this file has changed. A change seen
  /// before the run stops the call without running anything; one seen only
  /// after the run discards its result and the orbitals it wrote. On Linux,
  /// FreeBSD and NetBSD the call checks an open descriptor of the file and
  /// runs that descriptor with fexecve, so a file swapped in after the
  /// check is not run. Elsewhere, or when the kernel cannot run a
  /// descriptor, the child runs the path with execv and a swap in that
  /// window is caught only by the check after the run.
  [[nodiscard]] const std::string &resolvedExecutablePath() const noexcept {
    return m_real_executable;
  }

  /// Directory holding the input, output and .gbw files of the instance.
  [[nodiscard]] const std::string &runDirectory() const noexcept {
    return m_run_dir;
  }

  /// True when the last successful call started from stored orbitals.
  [[nodiscard]] bool reusedWavefunction() const noexcept {
    return m_last_reused;
  }

  /// True when an executable can be resolved for @p executable (empty for
  /// `orca` on PATH).
  [[nodiscard]] static bool probeAvailable(const std::string &executable = {});

  static constexpr uint64_t kKernelVersion = 2;

private:
  ORCAConfig m_config;
  std::string m_executable;
  std::string m_real_executable;
  uint64_t m_executable_stamp = 0;
  std::string m_run_dir;
  mutable bool m_has_guess = false;
  mutable bool m_last_reused = false;
  mutable std::vector<int> m_guess_numbers;
};

} // namespace rgpot
