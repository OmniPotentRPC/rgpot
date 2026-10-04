#pragma once
// MIT License — the system contract a UMA AOTI package embeds

#include "rgpot/UmaPot/UmaConfig.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rgpot {

/**
 * @brief A UMA package and the calculator disagree on the system.
 *
 * ``merge_mole`` folds one composition, charge and spin into the
 * compiled graph. Evaluating it for anything else either aborts inside
 * the graph (composition) or silently returns a different potential
 * energy surface (charge, spin, task), so UmaPot refuses instead.
 */
class UmaContractError : public std::runtime_error {
public:
  UmaContractError(std::string field, std::string package, std::string actual,
                   const std::string &where)
      : std::runtime_error("UmaPot: " + where + " " + field + " is " + actual +
                           ", the package was exported for " + package),
        m_field(std::move(field)), m_package(std::move(package)),
        m_actual(std::move(actual)) {}

  /// Metadata key that disagrees (``charge``, ``spin``, ``task_name``,
  /// ``z_set``, ``counts`` or ``natoms``).
  [[nodiscard]] const std::string &field() const noexcept { return m_field; }
  /// Value embedded in the package.
  [[nodiscard]] const std::string &package() const noexcept {
    return m_package;
  }
  /// Value the config or the input carries.
  [[nodiscard]] const std::string &actual() const noexcept { return m_actual; }

private:
  std::string m_field;
  std::string m_package;
  std::string m_actual;
};

/**
 * @brief Charge, spin, task and composition embedded in a ``.pt2``.
 *
 * Built from ``AOTIModelPackageLoader::get_metadata()``. Every field is
 * optional: a key the exporter did not write is not checked.
 * ``scripts/export_uma_aoti.py`` writes ``task_name``, ``charge``,
 * ``spin``, ``z_set`` (a list such as ``[1, 6, 7]``), ``natoms`` and
 * ``counts`` (a JSON object of atomic number to atom count, such as
 * ``{"1": 1, "6": 1, "7": 1}``), plus ``nedges_min`` and ``nedges_max``,
 * the edge-count range the exported graph accepts (equal for a static
 * package, whose graph is traced at one edge count); packages from older
 * exporters carry the first four only.
 */
struct UmaContract {
  std::optional<std::string> task_name;
  std::optional<int64_t> charge;
  std::optional<int64_t> spin;
  std::optional<std::vector<int64_t>> z_set;
  std::optional<std::map<int64_t, int64_t>> counts;
  std::optional<int64_t> natoms;
  std::optional<int64_t> nedges_min;
  std::optional<int64_t> nedges_max;

  /// Signed decimal integers in order of appearance; anything else
  /// separates them.
  static std::vector<int64_t> integers(const std::string &s) {
    std::vector<int64_t> out;
    size_t i = 0;
    while (i < s.size()) {
      const bool neg =
          s[i] == '-' && i + 1 < s.size() && s[i + 1] >= '0' && s[i + 1] <= '9';
      if (!neg && !(s[i] >= '0' && s[i] <= '9')) {
        ++i;
        continue;
      }
      size_t j = neg ? i + 1 : i;
      int64_t v = 0;
      while (j < s.size() && s[j] >= '0' && s[j] <= '9')
        v = v * 10 + (s[j++] - '0');
      out.push_back(neg ? -v : v);
      i = j;
    }
    return out;
  }

  static UmaContract
  fromMetadata(const std::unordered_map<std::string, std::string> &meta) {
    auto get = [&](const char *key) -> const std::string * {
      const auto it = meta.find(key);
      return it == meta.end() || it->second.empty() ? nullptr : &it->second;
    };
    auto one_int = [&](const char *key) -> std::optional<int64_t> {
      const std::string *raw = get(key);
      if (raw == nullptr)
        return std::nullopt;
      const auto v = integers(*raw);
      if (v.size() != 1)
        throw std::runtime_error(std::string("UmaPot: package metadata ") +
                                 key + " is not one integer: " + *raw);
      return v.front();
    };
    UmaContract c;
    if (const std::string *raw = get("task_name"))
      c.task_name = *raw;
    c.charge = one_int("charge");
    c.spin = one_int("spin");
    c.natoms = one_int("natoms");
    c.nedges_min = one_int("nedges_min");
    c.nedges_max = one_int("nedges_max");
    if (const std::string *raw = get("z_set")) {
      auto zs = integers(*raw);
      std::sort(zs.begin(), zs.end());
      c.z_set = std::move(zs);
    }
    if (const std::string *raw = get("counts")) {
      const auto v = integers(*raw);
      if (v.size() % 2 != 0)
        throw std::runtime_error(
            "UmaPot: package metadata counts is not {Z: n} pairs: " + *raw);
      std::map<int64_t, int64_t> m;
      for (size_t k = 0; k < v.size(); k += 2)
        m[v[k]] = v[k + 1];
      c.counts = std::move(m);
    }
    return c;
  }

  static std::string str(const std::vector<int64_t> &v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i)
      s += (i ? ", " : "") + std::to_string(v[i]);
    return s + "]";
  }

  static std::string str(const std::map<int64_t, int64_t> &m) {
    std::string s = "{";
    bool first = true;
    for (const auto &[z, n] : m) {
      s += (first ? "" : ", ") + std::to_string(z) + ": " + std::to_string(n);
      first = false;
    }
    return s + "}";
  }

  /// Task, charge and spin of the calculator against the package.
  void checkConfig(const UmaConfig &cfg) const {
    if (task_name && *task_name != cfg.task_name)
      throw UmaContractError("task_name", *task_name, cfg.task_name, "config");
    if (charge && *charge != cfg.charge)
      throw UmaContractError("charge", std::to_string(*charge),
                             std::to_string(cfg.charge), "config");
    if (spin && *spin != cfg.spin)
      throw UmaContractError("spin", std::to_string(*spin),
                             std::to_string(cfg.spin), "config");
  }

  /// Edge count of one call (the whole band for a batched package)
  /// against the range the compiled graph was exported for.
  void checkEdges(int64_t nedges) const {
    const bool below = nedges_min && nedges < *nedges_min;
    const bool above = nedges_max && nedges > *nedges_max;
    if (!below && !above)
      return;
    const int64_t lo = nedges_min ? *nedges_min : 0;
    const std::string want =
        nedges_max && lo == *nedges_max
            ? std::to_string(lo)
            : "[" + std::to_string(lo) + ", " +
                  (nedges_max ? std::to_string(*nedges_max) : "inf") + "]";
    throw UmaContractError("nedges", want, std::to_string(nedges), "input");
  }

  /// Atom count, element set and per-element counts of one system.
  void checkSystem(size_t nAtoms, const int *atmnrs) const {
    if (natoms && *natoms != static_cast<int64_t>(nAtoms))
      throw UmaContractError("natoms", std::to_string(*natoms),
                             std::to_string(nAtoms), "input");
    if (!z_set && !counts)
      return;
    std::map<int64_t, int64_t> have;
    for (size_t i = 0; i < nAtoms; ++i)
      ++have[atmnrs[i]];
    if (counts && *counts != have)
      throw UmaContractError("counts", str(*counts), str(have), "input");
    if (z_set) {
      std::vector<int64_t> zs;
      zs.reserve(have.size());
      for (const auto &kv : have)
        zs.push_back(kv.first);
      if (zs != *z_set)
        throw UmaContractError("z_set", str(*z_set), str(zs), "input");
    }
  }
};

} // namespace rgpot
