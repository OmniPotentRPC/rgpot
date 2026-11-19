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

} // namespace rgpot::engine
