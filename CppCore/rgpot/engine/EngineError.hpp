#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <stdexcept>
#include <string>

namespace rgpot::engine {

/// A backend failure whose C ABI return code is part of the failure, rather
/// than inferred from its diagnostic text.
class BackendError : public std::runtime_error {
public:
  BackendError(int code, const std::string &message)
      : std::runtime_error(message), code_(code) {}

  [[nodiscard]] int code() const noexcept { return code_; }

private:
  int code_;
};

/// Thrown by a setting hook for a value the backend cannot take; the adapter
/// returns 3 with the reason in last_error.
struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};

} // namespace rgpot::engine
