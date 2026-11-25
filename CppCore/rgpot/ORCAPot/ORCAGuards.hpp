#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file ORCAGuards.hpp
 * @brief Checks both ORCA frontends make before ORCA runs.
 *
 * ORCAPot, over liborcac, and ORCALegacyPot, over a child process, share
 * the box rule for an isolated cluster and the fingerprint of the orca
 * executable, so the two accept the same boxes and pin the file they run
 * the same way.
 */

#include <cstddef>
#include <cstdint>
#include <string>

namespace rgpot::orca {

/// Stamps the regular file open on @p fd with its size, modification time,
/// change time, device, inode and content hash into @p stamp. The change
/// time moves on every write, rename or attribute change and cannot be set
/// back by the owner, so a rewrite that restores the size and the
/// modification time still changes the stamp. False when the file cannot
/// be read. POSIX only.
#ifndef _WIN32
bool stampOpenFile(int fd, uint64_t *stamp) noexcept;
#endif

/// Resolves @p executable with symlinks followed into @p real, and stamps
/// the file it names with stampOpenFile into @p stamp. On Windows, where
/// neither ORCA frontend runs ORCA, the stamp holds only the size,
/// modification time, device and inode. False when the file cannot be
/// resolved, opened or read.
bool identifyExecutable(const std::string &executable, std::string *real,
                        uint64_t *stamp) noexcept;

/// True while @p executable still resolves to @p real with @p stamp.
bool executableUnchanged(const std::string &executable, const std::string &real,
                         uint64_t stamp) noexcept;

/// A box is taken as padding around an isolated cluster of @p n atoms at
/// @p pos (Angstrom). It passes when it is absent or all zero, or when
/// every pair of atoms is its own nearest periodic image: along every
/// nonzero lattice vector, not only the box vectors, the atoms span at most
/// half its length. Otherwise, and for a singular box, a cell too skewed to
/// check (including one whose image bound is not finite or out of range), a
/// non-finite entry or, with a box, a non-finite position, it throws
/// std::invalid_argument with a message that starts with @p who. The
/// positions are not moved.
void checkClusterBox(const char *who, const double *box, std::size_t n,
                     const double *pos);

} // namespace rgpot::orca
