#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file ORCAError.hpp
 * @brief The exception both ORCA frontends throw for a failed ORCA run.
 */

#include <stdexcept>

namespace rgpot {

/// A failure of the ORCA run itself: the executable changed since the
/// instance was made or while it ran, ORCA could not be started, ran past
/// its timeout, ended with an error, reported an SCF that did not converge,
/// or left results that cannot be read. The child-process runner's message
/// carries the tail of ORCA's stderr and output. ORCAPot also throws it when
/// liborcac is not loaded or returns a nonzero status. Input a caller got
/// wrong (a box in which the atoms would wrap, an unknown atomic number, an
/// impossible spin, a timeout that is negative or not finite) throws
/// std::invalid_argument instead.
class ORCAError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

} // namespace rgpot
