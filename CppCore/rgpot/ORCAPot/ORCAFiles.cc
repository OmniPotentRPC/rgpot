// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ORCAPot/ORCAFiles.hpp"

#include "rgpot/elements.hpp"

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <locale>
#include <locale.h>
#if defined(__APPLE__) || defined(__FreeBSD__)
#include <xlocale.h>
#endif
#include <sstream>
#include <stdexcept>

namespace rgpot::orca {

namespace {

void rejectLineBreak(const std::string &field, const char *name) {
  if (field.find_first_of("\r\n") != std::string::npos) {
    throw std::invalid_argument(std::string("ORCALegacyPot: ") + name +
                                " must not contain a line break");
  }
}

void appendWord(std::string &line, const std::string &word) {
  if (word.empty()) {
    return;
  }
  line += ' ';
  line += word;
}

[[noreturn]] void malformed(const std::string &what) {
  throw std::runtime_error("ORCALegacyPot: malformed .engrad file: " + what);
}

locale_t cLocale() {
  static const locale_t loc = ::newlocale(LC_NUMERIC_MASK, "C", nullptr);
  if (loc == nullptr) {
    throw std::runtime_error("ORCALegacyPot: could not create the C locale");
  }
  return loc;
}

double parseReal(const std::string &token, const char *what) {
  errno = 0;
  char *end = nullptr;
  const double value = ::strtod_l(token.c_str(), &end, cLocale());
  if (end == token.c_str() || *end != '\0') {
    malformed(std::string(what) + " '" + token + "' is not a number");
  }
  return value;
}

long parseInteger(const std::string &token, const char *what) {
  errno = 0;
  char *end = nullptr;
  const long value = ::strtol_l(token.c_str(), &end, 10, cLocale());
  if (end == token.c_str() || *end != '\0' || errno == ERANGE) {
    malformed(std::string(what) + " '" + token + "' is not an integer");
  }
  return value;
}

} // namespace

std::string elementSymbol(int z) {
  if (z < 1 || z > static_cast<int>(elements::symbols.size())) {
    throw std::invalid_argument("ORCALegacyPot: atomic number " + std::to_string(z) +
                                " has no element symbol");
  }
  return std::string(elements::symbols[static_cast<std::size_t>(z) - 1]);
}

std::string renderInput(const InputSpec &spec, std::size_t n_atoms,
                        const double *pos_angstrom,
                        const int *atomic_numbers) {
  rejectLineBreak(spec.method, "method");
  rejectLineBreak(spec.basis, "basis");
  rejectLineBreak(spec.keywords, "keywords");
  rejectLineBreak(spec.moinp, "moinp");

  std::string text = "!";
  appendWord(text, spec.method);
  appendWord(text, spec.basis);
  appendWord(text, spec.keywords);
  text += " EnGrad";
  if (!spec.moinp.empty()) {
    text += " MORead";
  }
  text += '\n';
  std::ostringstream out;
  out.imbue(std::locale::classic());
  if (spec.maxcore_mb > 0) {
    out << "%maxcore " << spec.maxcore_mb << '\n';
  }
  if (spec.nprocs > 1) {
    out << "%pal nprocs " << spec.nprocs << " end\n";
  }
  out << "%scf ConvForced true end\n";
  if (!spec.moinp.empty()) {
    out << "%moinp \"" << spec.moinp << "\"\n";
  }
  out << "* xyz " << spec.charge << ' ' << spec.multiplicity << '\n';
  out << std::fixed << std::setprecision(12);
  for (std::size_t i = 0; i < n_atoms; ++i) {
    const std::string symbol = elementSymbol(atomic_numbers[i]);
    out << symbol << (symbol.size() < 2 ? "  " : " ") << pos_angstrom[3 * i]
        << ' ' << pos_angstrom[3 * i + 1] << ' ' << pos_angstrom[3 * i + 2]
        << '\n';
  }
  out << "*\n";
  text += out.str();
  return text;
}

Engrad parseEngrad(const std::string &text, std::size_t n_atoms) {
  std::vector<std::string> tokens;
  std::istringstream lines(text);
  lines.imbue(std::locale::classic());
  std::string line;
  while (std::getline(lines, line)) {
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') {
      continue;
    }
    std::istringstream words(line);
    words.imbue(std::locale::classic());
    std::string word;
    while (words >> word) {
      tokens.push_back(word);
    }
  }
  if (tokens.empty()) {
    malformed("no data");
  }
  const long declared = parseInteger(tokens[0], "the atom count");
  if (declared < 0 || static_cast<std::size_t>(declared) != n_atoms) {
    malformed("it holds " + tokens[0] + " atoms, the input had " +
              std::to_string(n_atoms));
  }
  const std::size_t expected = 2 + 3 * n_atoms + 4 * n_atoms;
  if (tokens.size() != expected) {
    malformed("expected " + std::to_string(expected) + " values, found " +
              std::to_string(tokens.size()));
  }
  Engrad out;
  out.energy_hartree = parseReal(tokens[1], "the energy");
  out.gradient_hartree_per_bohr.reserve(3 * n_atoms);
  std::size_t at = 2;
  for (std::size_t k = 0; k < 3 * n_atoms; ++k) {
    out.gradient_hartree_per_bohr.push_back(
        parseReal(tokens[at++], "a gradient component"));
  }
  out.atomic_numbers.reserve(n_atoms);
  for (std::size_t i = 0; i < n_atoms; ++i) {
    out.atomic_numbers.push_back(
        static_cast<int>(parseInteger(tokens[at], "an atomic number")));
    // The Bohr coordinates that follow are not used.
    at += 4;
  }
  return out;
}

std::string fileTail(const std::string &path, std::size_t max_bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return "(" + path + " could not be read)";
  }
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size <= 0) {
    return "(empty)";
  }
  const std::streamoff keep =
      size < static_cast<std::streamoff>(max_bytes)
          ? size
          : static_cast<std::streamoff>(max_bytes);
  in.seekg(size - keep, std::ios::beg);
  std::string tail(static_cast<std::size_t>(keep), '\0');
  in.read(tail.data(), keep);
  tail.resize(static_cast<std::size_t>(in.gcount()));
  if (keep < size) {
    const auto cut = tail.find('\n');
    if (cut != std::string::npos && cut + 1 < tail.size()) {
      tail.erase(0, cut + 1);
    }
  }
  while (!tail.empty() && (tail.back() == '\n' || tail.back() == '\r')) {
    tail.pop_back();
  }
  return tail.empty() ? "(empty)" : tail;
}

} // namespace rgpot::orca
