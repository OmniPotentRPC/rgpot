// MIT License
// Copyright 2023--present rgpot developers

#include "NblibTopology.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <map>
#include <sstream>
#include <system_error>

namespace rgpot::nblibpot {

namespace {

[[noreturn]] void fail(std::size_t line, const std::string &what) {
  throw std::invalid_argument("nblib topology line " + std::to_string(line) +
                              ": " + what);
}

double number(std::size_t line, const std::string &token, const char *what) {
  double value = 0.0;
  const char *first = token.data();
  const char *last = first + token.size();
  const auto result = std::from_chars(first, last, value);
  if (result.ec != std::errc() || result.ptr != last || !std::isfinite(value)) {
    fail(line, std::string(what) + " is not a finite number: " + token);
  }
  return value;
}

int whole(std::size_t line, const std::string &token, const char *what) {
  int value = 0;
  const char *first = token.data();
  const char *last = first + token.size();
  const auto result = std::from_chars(first, last, value);
  if (result.ec != std::errc() || result.ptr != last) {
    fail(line, std::string(what) + " is not an integer: " + token);
  }
  return value;
}

int atomIndex(std::size_t line, const MoleculeDesc &molecule,
              const std::string &name) {
  for (std::size_t i = 0; i < molecule.atoms.size(); ++i) {
    if (molecule.atoms[i].name == name) {
      return static_cast<int>(i);
    }
  }
  fail(line, "molecule " + molecule.name + " has no atom " + name);
}

void needTokens(std::size_t line, const std::vector<std::string> &tokens,
                std::size_t count, const char *usage) {
  if (tokens.size() != count) {
    fail(line, std::string("expected ") + usage);
  }
}

ListedKind listedKind(std::size_t line, const std::string &family,
                      const std::string &form) {
  if (family == "bond" && form == "harmonic") {
    return ListedKind::HarmonicBond;
  }
  if (family == "bond" && form == "g96") {
    return ListedKind::G96Bond;
  }
  if (family == "bond" && form == "morse") {
    return ListedKind::MorseBond;
  }
  if (family == "angle" && form == "harmonic") {
    return ListedKind::HarmonicAngle;
  }
  if (family == "angle" && form == "g96") {
    return ListedKind::G96Angle;
  }
  if (family == "dihedral" && form == "proper") {
    return ListedKind::ProperDihedral;
  }
  if (family == "dihedral" && form == "improper") {
    return ListedKind::ImproperDihedral;
  }
  if (family == "dihedral" && form == "rb") {
    return ListedKind::RyckaertBelleman;
  }
  fail(line, "unknown listed interaction " + family + " " + form);
}

void addListed(std::size_t line, MoleculeDesc &molecule, ListedKind kind,
               const std::vector<std::string> &tokens, std::size_t first) {
  const std::size_t atoms = centres(kind);
  const std::size_t params = parameterCount(kind);
  if (tokens.size() != first + atoms + params) {
    fail(line, "expected " + std::to_string(atoms) + " atom names and " +
                   std::to_string(params) + " parameters");
  }
  ListedDesc term;
  term.kind = kind;
  for (std::size_t i = 0; i < atoms; ++i) {
    term.atoms.push_back(atomIndex(line, molecule, tokens[first + i]));
  }
  for (std::size_t i = 0; i < atoms; ++i) {
    for (std::size_t j = i + 1; j < atoms; ++j) {
      if (term.atoms[i] == term.atoms[j]) {
        fail(line, "an interaction names atom " + tokens[first + i] + " twice");
      }
    }
  }
  for (std::size_t i = 0; i < params; ++i) {
    term.params.push_back(number(line, tokens[first + atoms + i], "parameter"));
  }
  if (kind == ListedKind::ProperDihedral) {
    const double multiplicity = term.params[2];
    if (multiplicity < 1.0 || std::floor(multiplicity) != multiplicity) {
      fail(line, "dihedral multiplicity must be a positive integer");
    }
  }
  molecule.listed.push_back(std::move(term));
}

} // namespace

std::size_t centres(ListedKind kind) {
  switch (kind) {
  case ListedKind::HarmonicBond:
  case ListedKind::G96Bond:
  case ListedKind::MorseBond:
  case ListedKind::Pair14:
    return 2;
  case ListedKind::HarmonicAngle:
  case ListedKind::G96Angle:
    return 3;
  case ListedKind::ProperDihedral:
  case ListedKind::ImproperDihedral:
  case ListedKind::RyckaertBelleman:
    return 4;
  }
  throw std::logic_error("unhandled listed kind");
}

std::size_t parameterCount(ListedKind kind) {
  switch (kind) {
  case ListedKind::HarmonicBond:
  case ListedKind::G96Bond:
  case ListedKind::Pair14:
  case ListedKind::HarmonicAngle:
  case ListedKind::G96Angle:
  case ListedKind::ImproperDihedral:
    return 2;
  case ListedKind::MorseBond:
  case ListedKind::ProperDihedral:
    return 3;
  case ListedKind::RyckaertBelleman:
    return 6;
  }
  throw std::logic_error("unhandled listed kind");
}

std::size_t TopologyDesc::atomCount() const {
  std::size_t total = 0;
  for (const MoleculeDesc &molecule : molecules) {
    total += molecule.atoms.size() * static_cast<std::size_t>(molecule.count);
  }
  return total;
}

std::vector<const AtomTypeDesc *> TopologyDesc::atomTypes() const {
  std::map<std::string, const AtomTypeDesc *> byName;
  for (const AtomTypeDesc &type : types) {
    byName[type.name] = &type;
  }
  std::vector<const AtomTypeDesc *> out;
  out.reserve(atomCount());
  for (const MoleculeDesc &molecule : molecules) {
    for (int copy = 0; copy < molecule.count; ++copy) {
      for (const AtomDesc &atom : molecule.atoms) {
        out.push_back(byName.at(atom.type));
      }
    }
  }
  return out;
}

std::vector<double> TopologyDesc::charges() const {
  std::vector<double> out;
  out.reserve(atomCount());
  for (const MoleculeDesc &molecule : molecules) {
    for (int copy = 0; copy < molecule.count; ++copy) {
      for (const AtomDesc &atom : molecule.atoms) {
        out.push_back(coulomb == CoulombMode::None ? 0.0 : atom.charge);
      }
    }
  }
  return out;
}

std::vector<std::array<std::size_t, 2>> TopologyDesc::instances() const {
  std::vector<std::array<std::size_t, 2>> out;
  std::size_t first = 0;
  for (const MoleculeDesc &molecule : molecules) {
    for (int copy = 0; copy < molecule.count; ++copy) {
      out.push_back({first, molecule.atoms.size()});
      first += molecule.atoms.size();
    }
  }
  return out;
}

TopologyDesc parseTopology(std::string_view text) {
  TopologyDesc topology;
  std::istringstream input{std::string(text)};
  std::string raw;
  std::size_t line = 0;
  MoleculeDesc *open = nullptr;
  bool haveCutoff = false;
  while (std::getline(input, raw)) {
    ++line;
    const std::size_t hash = raw.find('#');
    if (hash != std::string::npos) {
      raw.erase(hash);
    }
    std::istringstream words(raw);
    std::vector<std::string> tokens;
    for (std::string word; words >> word;) {
      tokens.push_back(word);
    }
    if (tokens.empty()) {
      continue;
    }
    const std::string &key = tokens[0];
    if (key == "nblib") {
      needTokens(line, tokens, 2, "nblib 1");
      if (whole(line, tokens[1], "format version") != 1) {
        fail(line, "the format version is 1");
      }
    } else if (key == "cutoff") {
      needTokens(line, tokens, 2, "cutoff <nm>");
      topology.cutoff = number(line, tokens[1], "cutoff");
      if (!(topology.cutoff > 0.0)) {
        fail(line, "the cutoff must be positive");
      }
      haveCutoff = true;
    } else if (key == "coulomb") {
      needTokens(line, tokens, 2, "coulomb none|cutoff");
      if (tokens[1] == "none") {
        topology.coulomb = CoulombMode::None;
      } else if (tokens[1] == "cutoff") {
        topology.coulomb = CoulombMode::Cutoff;
      } else {
        fail(line, "coulomb is none or cutoff");
      }
    } else if (key == "atomtype") {
      if (open != nullptr) {
        fail(line, "atomtype inside a molecule");
      }
      needTokens(line, tokens, 5, "atomtype <name> <mass> <c6> <c12>");
      AtomTypeDesc type;
      type.name = tokens[1];
      type.mass = number(line, tokens[2], "mass");
      type.c6 = number(line, tokens[3], "c6");
      type.c12 = number(line, tokens[4], "c12");
      if (!(type.mass > 0.0)) {
        fail(line, "the mass must be positive");
      }
      for (const AtomTypeDesc &known : topology.types) {
        if (known.name == type.name) {
          fail(line, "atomtype " + type.name + " is defined twice");
        }
      }
      topology.types.push_back(std::move(type));
    } else if (key == "pair") {
      if (open != nullptr) {
        fail(line, "pair inside a molecule; use pair14 for bonded pairs");
      }
      needTokens(line, tokens, 5, "pair <typeA> <typeB> <c6> <c12>");
      PairDesc pair;
      pair.a = tokens[1];
      pair.b = tokens[2];
      pair.c6 = number(line, tokens[3], "c6");
      pair.c12 = number(line, tokens[4], "c12");
      topology.pairs.push_back(std::move(pair));
    } else if (key == "molecule") {
      if (open != nullptr) {
        fail(line, "molecule inside a molecule");
      }
      needTokens(line, tokens, 3, "molecule <name> <count>");
      MoleculeDesc molecule;
      molecule.name = tokens[1];
      molecule.count = whole(line, tokens[2], "count");
      if (molecule.count < 1) {
        fail(line, "the molecule count must be positive");
      }
      topology.molecules.push_back(std::move(molecule));
      open = &topology.molecules.back();
    } else if (key == "end") {
      needTokens(line, tokens, 1, "end");
      if (open == nullptr) {
        fail(line, "end outside a molecule");
      }
      if (open->atoms.empty()) {
        fail(line, "molecule " + open->name + " has no atoms");
      }
      open = nullptr;
    } else if (open == nullptr) {
      fail(line, "unknown directive " + key);
    } else if (key == "atom") {
      needTokens(line, tokens, 4, "atom <name> <type> <charge>");
      for (const AtomDesc &known : open->atoms) {
        if (known.name == tokens[1]) {
          fail(line, "atom " + tokens[1] + " is defined twice");
        }
      }
      bool typed = false;
      for (const AtomTypeDesc &type : topology.types) {
        typed = typed || type.name == tokens[2];
      }
      if (!typed) {
        fail(line, "atom " + tokens[1] + " has the undefined type " + tokens[2]);
      }
      open->atoms.push_back(
          {tokens[1], tokens[2], number(line, tokens[3], "charge")});
    } else if (key == "exclude") {
      needTokens(line, tokens, 3, "exclude <atomA> <atomB>");
      const int a = atomIndex(line, *open, tokens[1]);
      const int b = atomIndex(line, *open, tokens[2]);
      if (a == b) {
        fail(line, "an atom is not excluded from itself");
      }
      open->exclusions.push_back({a, b});
    } else if (key == "pair14") {
      addListed(line, *open, ListedKind::Pair14, tokens, 1);
    } else if (key == "bond" || key == "angle" || key == "dihedral") {
      if (tokens.size() < 2) {
        fail(line, key + " needs a form");
      }
      addListed(line, *open, listedKind(line, key, tokens[1]), tokens, 2);
    } else {
      fail(line, "unknown directive " + key);
    }
  }
  if (open != nullptr) {
    fail(line, "molecule " + open->name + " is not closed with end");
  }
  if (!haveCutoff) {
    throw std::invalid_argument("nblib topology: no cutoff directive");
  }
  if (topology.types.empty() || topology.molecules.empty()) {
    throw std::invalid_argument(
        "nblib topology: needs an atomtype and a molecule");
  }
  for (const PairDesc &pair : topology.pairs) {
    for (const std::string *name : {&pair.a, &pair.b}) {
      const bool known = std::any_of(
          topology.types.begin(), topology.types.end(),
          [&](const AtomTypeDesc &type) { return type.name == *name; });
      if (!known) {
        throw std::invalid_argument("nblib topology: pair names the undefined "
                                    "atomtype " +
                                    *name);
      }
    }
  }
  return topology;
}

} // namespace rgpot::nblibpot
