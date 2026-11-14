#pragma once
// MIT License — the system contract a UMA AOTI package embeds

#include "rgpot/UmaPot/UmaConfig.hpp"

#include <algorithm>
#include <cctype>
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
/**
 * @brief The positions or the edge list are not the shape this package
 * was compiled for.
 *
 * The message names both shapes. Callers can match ``shape is``.
 */
class UmaShapeError : public std::runtime_error {
public:
  UmaShapeError(std::string field, std::string actual, std::string package)
      : std::runtime_error("UmaPot: " + field + " shape is " + actual +
                           ", the package was exported for " + package),
        m_field(std::move(field)), m_actual(std::move(actual)),
        m_package(std::move(package)) {}

  [[nodiscard]] const std::string &field() const noexcept { return m_field; }
  [[nodiscard]] const std::string &actual() const noexcept { return m_actual; }
  [[nodiscard]] const std::string &package() const noexcept {
    return m_package;
  }

private:
  std::string m_field;
  std::string m_actual;
  std::string m_package;
};

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
 *
 * A package with neither edge key takes the edge count from the shapes
 * the graph was compiled for (compiledInputs), then from ``shapes``.
 */
struct UmaContract {
  std::optional<std::string> task_name;
  std::optional<int64_t> charge;
  std::optional<int64_t> spin;
  std::optional<std::vector<int64_t>> z_set;
  std::optional<std::map<int64_t, int64_t>> counts;
  std::optional<int64_t> natoms;
  std::optional<int64_t> natoms_min;
  std::optional<int64_t> natoms_max;
  std::optional<int64_t> nedges_min;
  std::optional<int64_t> nedges_max;

  /// One axis of a declared tensor shape. A symbolic axis is a name
  /// such as ``nedges`` with an inclusive range; a fixed axis is one
  /// integer.
  struct Axis {
    bool symbolic{false};
    int64_t fixed{0};
    int64_t lo{0};
    int64_t hi{0};
    std::string name;
  };

  struct ShapeDecl {
    bool known{false};
    std::vector<Axis> axes;
    std::string text;
  };

  ShapeDecl pos_decl;
  ShapeDecl edge_decl;
  /// Compiled ``pos`` of one call, used only when the metadata declares
  /// no atom count. A band call stacks its systems, so the rows are the
  /// atoms of every system in the chunk.
  ShapeDecl compiled_pos;
  /// Where the edge range came from: ``metadata``, ``compiled`` or
  /// ``shapes``.
  std::string edge_source{"metadata"};

  /// Input shapes the AOTI wrapper source checks. Each graph input has a
  /// ``check_input_<i>`` that compares a fixed axis
  /// (``if (3 != arg0_1_size[1])``) and bounds a symbolic one
  /// (``if (arg4_1_size[1] < 2)``, ``if (arg4_1_size[1] > 4096)``). Entry
  /// ``i`` is graph input ``i``. An input without such a function is not
  /// known, and an axis with no bound is open on that side.
  static std::vector<ShapeDecl> compiledInputs(const std::string &src) {
    std::vector<ShapeDecl> out;
    const std::string fn = "static void check_input_";
    const std::string handle = "= ConstantHandle(input_handles[";
    for (std::size_t at = src.find(fn); at != std::string::npos;
         at = src.find(fn, at + fn.size())) {
      // The generated function closes with a brace in column zero.
      const std::size_t stop = src.find("\n}", at);
      const std::string body =
          src.substr(at, (stop == std::string::npos ? src.size() : stop) - at);
      const std::size_t h = body.find(handle);
      if (h == std::string::npos)
        continue;
      const auto index = integers(body.substr(h + handle.size(), 24));
      if (index.empty() || index.front() < 0)
        continue;
      std::size_t v_end = h;
      while (v_end > 0 && body[v_end - 1] == ' ')
        --v_end;
      std::size_t v_begin = v_end;
      while (v_begin > 0 &&
             (std::isalnum(static_cast<unsigned char>(body[v_begin - 1])) ||
              body[v_begin - 1] == '_'))
        --v_begin;
      const std::string var = body.substr(v_begin, v_end - v_begin);
      if (var.empty())
        continue;
      const std::string size_ref = var + "_size[";
      const std::string stride_ref = var + "_stride[";

      struct Bound {
        bool fixed{false};
        int64_t value{0};
        int64_t lo{0};
        int64_t hi{INT64_MAX};
      };
      std::map<std::size_t, Bound> bounds;
      std::size_t rank = 0;
      auto axis_of = [](const std::string &tok,
                        const std::string &ref) -> std::optional<std::size_t> {
        if (tok.rfind(ref, 0) != 0 || tok.back() != ']')
          return std::nullopt;
        const auto d = integers(tok.substr(ref.size()));
        if (d.size() != 1 || d.front() < 0)
          return std::nullopt;
        return static_cast<std::size_t>(d.front());
      };
      auto literal = [](const std::string &tok) -> std::optional<int64_t> {
        if (tok.empty() || !std::all_of(tok.begin(), tok.end(), [](char ch) {
              return ch >= '0' && ch <= '9';
            }))
          return std::nullopt;
        return integers(tok).front();
      };
      for (std::size_t c = body.find("if ("); c != std::string::npos;
           c = body.find("if (", c + 4)) {
        const std::size_t close = body.find(')', c);
        if (close == std::string::npos)
          break;
        std::vector<std::string> tok;
        std::string cur;
        for (char ch : body.substr(c + 4, close - c - 4)) {
          if (ch == ' ') {
            if (!cur.empty())
              tok.push_back(cur);
            cur.clear();
          } else {
            cur.push_back(ch);
          }
        }
        if (!cur.empty())
          tok.push_back(cur);
        if (tok.size() != 3)
          continue;
        if (const auto d = axis_of(tok[2], stride_ref))
          rank = std::max(rank, *d + 1);
        if (const auto d = axis_of(tok[0], stride_ref))
          rank = std::max(rank, *d + 1);
        if (const auto d = axis_of(tok[2], size_ref)) {
          rank = std::max(rank, *d + 1);
          const auto k = literal(tok[0]);
          if (tok[1] == "!=" && k) {
            bounds[*d].fixed = true;
            bounds[*d].value = *k;
          }
          continue;
        }
        if (const auto d = axis_of(tok[0], size_ref)) {
          rank = std::max(rank, *d + 1);
          const auto k = literal(tok[2]);
          if (tok[1] == "<" && k)
            bounds[*d].lo = *k;
          else if (tok[1] == ">" && k)
            bounds[*d].hi = *k;
        }
      }
      if (rank == 0)
        continue;
      ShapeDecl decl;
      decl.known = true;
      for (std::size_t d = 0; d < rank; ++d) {
        const Bound b = bounds.count(d) ? bounds[d] : Bound{};
        Axis axis;
        if (b.fixed) {
          axis.fixed = b.value;
        } else {
          axis.symbolic = true;
          axis.lo = b.lo;
          axis.hi = b.hi;
        }
        decl.axes.push_back(axis);
      }
      const auto i = static_cast<std::size_t>(index.front());
      if (out.size() <= i)
        out.resize(i + 1);
      out[i] = std::move(decl);
    }
    return out;
  }

  static std::string bound(int64_t v) {
    return v == INT64_MAX ? "inf" : std::to_string(v);
  }

  /// Compiled input ``index`` when it is two-dimensional and axis
  /// ``fixed_axis`` is the constant ``value`` (``pos`` is ``[n, 3]``,
  /// ``edge_index`` is ``[2, e]``).
  static const ShapeDecl *compiledInput(const std::vector<ShapeDecl> &compiled,
                                        std::size_t index,
                                        std::size_t fixed_axis, int64_t value) {
    if (index >= compiled.size())
      return nullptr;
    const ShapeDecl &decl = compiled[index];
    if (!decl.known || decl.axes.size() != 2 ||
        decl.axes[fixed_axis].symbolic || decl.axes[fixed_axis].fixed != value)
      return nullptr;
    return &decl;
  }

  /// Second extent of ``edge_index`` inside the exporter's ``shapes``
  /// dict (``{'edge_index': [2, 1332], ...}``). Packages written before
  /// ``nedges_min`` and ``nedges_max`` record a fixed edge list there.
  /// It is the traced example, so the compiled edge_index wins when the
  /// package has one. A missing or non-fixed entry is not a count.
  static std::optional<int64_t>
  fixedEdgesFromShapes(const std::string *shapes) {
    if (shapes == nullptr)
      return std::nullopt;
    const std::string &s = *shapes;
    const std::string key = "edge_index";
    std::size_t at = 0;
    while (at < s.size()) {
      const auto found = s.find(key, at);
      if (found == std::string::npos)
        return std::nullopt;
      const bool boundary = found == 0 || s[found - 1] == '\'' ||
                            s[found - 1] == '"' || s[found - 1] == '{' ||
                            s[found - 1] == ' ' || s[found - 1] == ',';
      std::size_t i = found + key.size();
      if (!boundary || i > s.size()) {
        at = found + 1;
        continue;
      }
      if (i < s.size() && (s[i] == '\'' || s[i] == '"'))
        ++i;
      // ``edge_index_shape`` and other longer names are not this key.
      if (i < s.size() &&
          ((s[i] >= 'A' && s[i] <= 'Z') || (s[i] >= 'a' && s[i] <= 'z') ||
           s[i] == '_')) {
        at = i;
        continue;
      }
      while (i < s.size() && (s[i] == ' ' || s[i] == ':' || s[i] == '='))
        ++i;
      if (i >= s.size() || s[i] != '[')
        return std::nullopt;
      const auto close = s.find(']', i);
      if (close == std::string::npos)
        return std::nullopt;
      const auto dims = integers(s.substr(i, close - i + 1));
      if (dims.size() == 2 && dims[0] == 2 && dims[1] > 0)
        return dims[1];
      return std::nullopt;
    }
    return std::nullopt;
  }

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

  /// ``compiled`` is compiledInputs() of the package's wrapper source,
  /// empty when the package carries none.
  static UmaContract
  fromMetadata(const std::unordered_map<std::string, std::string> &meta,
               const std::vector<ShapeDecl> &compiled = {}) {
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
    // A package exported with fixed shapes carries neither range key.
    // The compiled edge_index is what the graph accepts: a fixed axis is
    // the one edge count, a symbolic axis its bounds. ``shapes`` only
    // records the example, so it is used when the wrapper source is
    // missing. The neighbour list is not padded; a list of another
    // length is named before the compiled graph runs. One key without
    // the other is still not a range.
    const ShapeDecl *compiled_edges = compiledInput(compiled, 4, 0, 2);
    if (!c.nedges_min && !c.nedges_max) {
      if (compiled_edges) {
        const Axis &e = compiled_edges->axes[1];
        c.nedges_min = e.symbolic ? e.lo : e.fixed;
        c.nedges_max = e.symbolic ? e.hi : e.fixed;
        c.edge_source = "compiled";
      } else if (const auto fixed = fixedEdgesFromShapes(get("shapes"))) {
        c.nedges_min = fixed;
        c.nedges_max = fixed;
        c.edge_source = "shapes";
      } else {
        throw UmaContractError(
            "nedges", "nedges_min and nedges_max or a compiled edge_index",
            "absent", "package");
      }
    } else if (!c.nedges_min || !c.nedges_max) {
      const char *missing = c.nedges_min ? "nedges_max" : "nedges_min";
      throw UmaContractError(missing, "nedges_min and nedges_max", "absent",
                             "package");
    }
    c.natoms_min = one_int("natoms_min");
    c.natoms_max = one_int("natoms_max");
    c.pos_decl = c.parseShape(get("pos_shape"));
    c.edge_decl = c.parseShape(get("edge_index_shape"));
    if (!c.pos_decl.known && c.natoms) {
      c.pos_decl.known = true;
      c.pos_decl.axes.push_back(Axis{false, *c.natoms, 0, 0, {}});
      c.pos_decl.axes.push_back(Axis{false, 3, 0, 0, {}});
      c.pos_decl.text = "[" + std::to_string(*c.natoms) + ", 3]";
    }
    if (!c.pos_decl.known) {
      if (const ShapeDecl *pos = compiledInput(compiled, 0, 1, 3)) {
        c.compiled_pos = *pos;
        const Axis &n = pos->axes[0];
        c.compiled_pos.text = n.symbolic
                                  ? "[natoms, 3] with natoms in [" +
                                        bound(n.lo) + ", " + bound(n.hi) + "]"
                                  : "[" + std::to_string(n.fixed) + ", 3]";
      }
    }
    if (!c.edge_decl.known && (c.nedges_min || c.nedges_max)) {
      const int64_t lo = c.nedges_min ? *c.nedges_min : 0;
      const int64_t hi = c.nedges_max ? *c.nedges_max : lo;
      c.edge_decl.known = true;
      c.edge_decl.axes.push_back(Axis{false, 2, 0, 0, {}});
      if (c.nedges_min && c.nedges_max && lo == hi) {
        c.edge_decl.axes.push_back(Axis{false, lo, 0, 0, {}});
        c.edge_decl.text = "[2, " + std::to_string(lo) + "]";
      } else {
        c.edge_decl.axes.push_back(Axis{true, 0, lo, hi, "nedges"});
        c.edge_decl.text = "[2, nedges] with nedges in [" + std::to_string(lo) +
                           ", " + bound(hi) + "]";
      }
    }
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

  /// ``spec`` is ``2,6`` or ``2,nedges`` (brackets and spaces allowed).
  ShapeDecl parseShape(const std::string *spec) const {
    ShapeDecl decl;
    if (spec == nullptr)
      return decl;
    std::string token;
    std::vector<std::string> tokens;
    auto flush = [&]() {
      if (!token.empty()) {
        tokens.push_back(token);
        token.clear();
      }
    };
    for (char ch : *spec) {
      if (ch == ',' || ch == '[' || ch == ']' || ch == ' ')
        flush();
      else
        token.push_back(ch);
    }
    flush();
    if (tokens.empty())
      throw std::runtime_error("UmaPot: package shape is empty: " + *spec);
    decl.known = true;
    decl.text = "[";
    for (size_t i = 0; i < tokens.size(); ++i) {
      const std::string &tok = tokens[i];
      const bool number =
          !tok.empty() &&
          std::all_of(tok.begin(), tok.end(),
                      [](char ch) { return ch >= '0' && ch <= '9'; });
      Axis axis;
      if (number) {
        axis.fixed = std::stoll(tok);
      } else {
        axis.symbolic = true;
        axis.name = tok;
        if (tok == "nedges") {
          axis.lo = nedges_min ? *nedges_min : 0;
          axis.hi = nedges_max ? *nedges_max : axis.lo;
        } else if (tok == "natoms") {
          axis.lo = natoms_min ? *natoms_min : (natoms ? *natoms : 1);
          axis.hi = natoms_max ? *natoms_max : (natoms ? *natoms : axis.lo);
        } else {
          throw std::runtime_error(
              "UmaPot: package shape has an unknown dimension " + tok);
        }
      }
      decl.axes.push_back(axis);
      decl.text += (i ? ", " : "") + tok;
    }
    decl.text += "]";
    for (const Axis &axis : decl.axes) {
      if (!axis.symbolic)
        continue;
      decl.text += " with " + axis.name + " in [" + std::to_string(axis.lo) +
                   ", " + std::to_string(axis.hi) + "]";
    }
    return decl;
  }

  static std::string shapeText(const std::vector<int64_t> &dims) {
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i)
      s += (i ? ", " : "") + std::to_string(dims[i]);
    return s + "]";
  }

  void matchShape(const std::string &field, const ShapeDecl &decl,
                  const std::vector<int64_t> &actual) const {
    if (!decl.known)
      return;
    bool ok = actual.size() == decl.axes.size();
    for (size_t i = 0; ok && i < actual.size(); ++i) {
      const Axis &axis = decl.axes[i];
      if (axis.symbolic)
        ok = actual[i] >= axis.lo && actual[i] <= axis.hi;
      else
        ok = actual[i] == axis.fixed;
    }
    if (!ok)
      throw UmaShapeError(field, shapeText(actual), decl.text);
  }

  /// Positions are ``[natoms, 3]`` and the edge list is ``[2, nedges]``.
  /// A static package names the traced sizes. A dynamic package names
  /// the symbolic axis and its range. Either way the message carries
  /// both shapes.
  void checkCallShapes(int64_t nAtoms, int64_t nedges) const {
    matchShape("pos", pos_decl, {nAtoms, 3});
    matchShape("edge_index", edge_decl, {2, nedges});
  }

  /// Rows of the ``pos`` tensor one call passes, against the compiled
  /// shape. Known only for a package whose metadata declares no atoms.
  void checkCompiledPos(int64_t rows) const {
    matchShape("pos", compiled_pos, {rows, 3});
  }

  /// Every system, on the single path and on the band path.
  ///
  /// An exact package stores the traced total, so each system must
  /// hold an equal share. A range package checks each system's own
  /// count against ``[nedges_min, nedges_max]`` instead of skipping
  /// the band. A molecular package is traced on every ordered pair,
  /// so each system must also have ``natoms * (natoms - 1)`` edges.
  void checkEachSystem(int64_t systems, const int64_t *counts,
                       int64_t natoms = 0, bool molecular = false) const {
    if (systems <= 0 || counts == nullptr)
      return;
    if (!nedges_min || !nedges_max)
      return;
    const bool exact = *nedges_min == *nedges_max;
    int64_t each_exact = 0;
    if (exact) {
      if (*nedges_min % systems != 0) {
        throw UmaContractError("nedges", std::to_string(*nedges_min),
                               "not divisible by the band", "package");
      }
      each_exact = *nedges_min / systems;
    }
    const int64_t complete = natoms > 1 ? natoms * (natoms - 1) : 0;
    for (int64_t s = 0; s < systems; ++s) {
      const int64_t count = counts[s];
      if (molecular && count != complete) {
        throw UmaShapeError("edge_index",
                            "[2, " + std::to_string(count) + "]",
                            "[2, " + std::to_string(complete) + "]");
      }
      if (exact) {
        if (count == each_exact)
          continue;
        throw UmaShapeError("edge_index",
                            "[2, " + std::to_string(count) + "]",
                            "[2, " + std::to_string(each_exact) + "]");
      }
      if (count < *nedges_min || count > *nedges_max) {
        throw UmaShapeError("edge_index", "[2, " + std::to_string(count) + "]",
                            "[2, nedges] with nedges in [" +
                                std::to_string(*nedges_min) + ", " +
                                bound(*nedges_max) + "]");
      }
    }
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
                  (nedges_max ? bound(*nedges_max) : "inf") + "]";
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
