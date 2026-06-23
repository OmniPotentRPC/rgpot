// MIT License
// Copyright 2023--present rgpot developers
//
// The unit expression parser in this file is derived from metatomic-torch
// (https://github.com/metatensor/metatomic), specifically PR #173 by
// Rohit Goswami. Original code is BSD-3-Clause licensed, copyright
// metatensor developers. Adapted for rgpot: removed torch/c10 dependency,
// using std::invalid_argument instead of C10_THROW_ERROR.

#include "rgpot/units.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace rgpot::units {

// ---- Dimensional analysis types ----

enum class DimIndex {
  LENGTH = 0,
  TIME = 1,
  MASS = 2,
  CHARGE = 3,
  TEMPERATURE = 4,
  COUNT = 5
};

struct Dimension {
  std::array<double, 5> exponents = {};

  Dimension operator*(const Dimension &other) const {
    Dimension result;
    for (size_t i = 0; i < static_cast<size_t>(DimIndex::COUNT); ++i) {
      result.exponents[i] = exponents[i] + other.exponents[i];
    }
    return result;
  }

  Dimension operator/(const Dimension &other) const {
    Dimension result;
    for (size_t i = 0; i < static_cast<size_t>(DimIndex::COUNT); ++i) {
      result.exponents[i] = exponents[i] - other.exponents[i];
    }
    return result;
  }

  Dimension pow(double p) const {
    Dimension result;
    for (size_t i = 0; i < static_cast<size_t>(DimIndex::COUNT); ++i) {
      result.exponents[i] = exponents[i] * p;
    }
    return result;
  }

  bool operator==(const Dimension &other) const {
    for (size_t i = 0; i < static_cast<size_t>(DimIndex::COUNT); ++i) {
      if (std::fabs(exponents[i] - other.exponents[i]) > 1e-10) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const Dimension &other) const { return !(*this == other); }

  std::string to_string() const {
    static const char *NAMES[] = {"L", "T", "M", "Q", "Th"};
    std::string result;
    bool first = true;
    for (size_t i = 0; i < static_cast<size_t>(DimIndex::COUNT); ++i) {
      double v = exponents[i];
      if (std::fabs(v) < 1e-10)
        continue;
      if (!first)
        result += " ";
      first = false;
      result += NAMES[i];
      if (std::fabs(v - 1.0) >= 1e-10 && std::fabs(v + 1.0) >= 1e-10) {
        if (std::fabs(v - std::round(v)) < 1e-10) {
          result += "^" + std::to_string(static_cast<int>(std::round(v)));
        } else {
          result += "^" + std::to_string(v);
        }
      }
      if (std::fabs(v + 1.0) < 1e-10) {
        result += "^-1";
      }
    }
    if (result.empty())
      result = "dimensionless";
    return result;
  }
};

struct UnitValue {
  double factor;
  Dimension dim;
};

//                                       L   T   M   Q   Th
static const Dimension DIM_LENGTH = {{1, 0, 0, 0, 0}};
static const Dimension DIM_TIME = {{0, 1, 0, 0, 0}};
static const Dimension DIM_MASS = {{0, 0, 1, 0, 0}};
static const Dimension DIM_CHARGE = {{0, 0, 0, 1, 0}};
static const Dimension DIM_ENERGY = {{2, -2, 1, 0, 0}};
static const Dimension DIM_NONE = {{0, 0, 0, 0, 0}};

static std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return c < 128 ? static_cast<char>(std::tolower(c))
                   : static_cast<char>(c);
  });
  return s;
}

// ---- Base unit table (SI factors) ----
// Derived from metatomic-torch units.cpp (metatensor/metatomic PR #173)

static const auto BASE_UNITS = std::unordered_map<std::string, UnitValue>{
    // Length
    {"angstrom", {1e-10, DIM_LENGTH}},
    {"a", {1e-10, DIM_LENGTH}},
    {"bohr", {5.2917721054482e-11, DIM_LENGTH}},
    {"nm", {1e-9, DIM_LENGTH}},
    {"nanometer", {1e-9, DIM_LENGTH}},
    {"meter", {1.0, DIM_LENGTH}},
    {"m", {1.0, DIM_LENGTH}},
    {"cm", {1e-2, DIM_LENGTH}},
    {"centimeter", {1e-2, DIM_LENGTH}},
    {"mm", {1e-3, DIM_LENGTH}},
    {"millimeter", {1e-3, DIM_LENGTH}},
    {"um", {1e-6, DIM_LENGTH}},
    {"\xc2\xb5m", {1e-6, DIM_LENGTH}}, // UTF-8 micro sign
    {"micrometer", {1e-6, DIM_LENGTH}},

    // Energy
    {"ev", {1.602176634e-19, DIM_ENERGY}},
    {"mev", {1.602176634e-19 * 1e-3, DIM_ENERGY}},
    {"hartree", {4.359744722206048e-18, DIM_ENERGY}},
    {"ry", {2.179872361103024e-18, DIM_ENERGY}},
    {"rydberg", {2.179872361103024e-18, DIM_ENERGY}},
    {"joule", {1.0, DIM_ENERGY}},
    {"j", {1.0, DIM_ENERGY}},
    {"kcal", {4184.0, DIM_ENERGY}},
    {"kj", {1000.0, DIM_ENERGY}},

    // Time
    {"s", {1.0, DIM_TIME}},
    {"second", {1.0, DIM_TIME}},
    {"ms", {1e-3, DIM_TIME}},
    {"millisecond", {1e-3, DIM_TIME}},
    {"us", {1e-6, DIM_TIME}},
    {"\xc2\xb5s", {1e-6, DIM_TIME}}, // UTF-8 micro sign
    {"microsecond", {1e-6, DIM_TIME}},
    {"ns", {1e-9, DIM_TIME}},
    {"nanosecond", {1e-9, DIM_TIME}},
    {"ps", {1e-12, DIM_TIME}},
    {"picosecond", {1e-12, DIM_TIME}},
    {"fs", {1e-15, DIM_TIME}},
    {"femtosecond", {1e-15, DIM_TIME}},

    // Mass
    {"u", {1.6605390689252e-27, DIM_MASS}},
    {"dalton", {1.6605390689252e-27, DIM_MASS}},
    {"kg", {1.0, DIM_MASS}},
    {"kilogram", {1.0, DIM_MASS}},
    {"g", {1e-3, DIM_MASS}},
    {"gram", {1e-3, DIM_MASS}},
    {"electron_mass", {9.109383713928e-31, DIM_MASS}},
    {"m_e", {9.109383713928e-31, DIM_MASS}},

    // Charge
    {"e", {1.602176634e-19, DIM_CHARGE}},
    {"coulomb", {1.0, DIM_CHARGE}},
    {"c", {1.0, DIM_CHARGE}},

    // Dimensionless
    {"mol", {6.02214076e23, DIM_NONE}},

    // Derived
    {"hbar", {1.0545718176462e-34, {{2, -1, 1, 0, 0}}}},
};

// ---- Tokenizer ----

enum class TokenType { LParen, RParen, Mul, Div, Pow, Value };

struct Token {
  TokenType type;
  std::string value;

  int precedence() const {
    switch (type) {
    case TokenType::LParen:
    case TokenType::RParen:
      return 0;
    case TokenType::Mul:
    case TokenType::Div:
      return 10;
    case TokenType::Pow:
      return 20;
    default:
      return -1;
    }
  }

  std::string as_str() const {
    switch (type) {
    case TokenType::LParen:
      return "(";
    case TokenType::RParen:
      return ")";
    case TokenType::Mul:
      return "*";
    case TokenType::Div:
      return "/";
    case TokenType::Pow:
      return "^";
    case TokenType::Value:
      return value;
    }
    return "?";
  }
};

static std::vector<Token> tokenize(const std::string &unit) {
  std::vector<Token> tokens;
  std::string current;
  for (size_t i = 0; i < unit.size(); ++i) {
    auto byte = static_cast<unsigned char>(unit[i]);
    // Handle UTF-8 micro sign (U+00B5): 0xC2 0xB5
    if (byte == 0xC2 && i + 1 < unit.size() &&
        static_cast<unsigned char>(unit[i + 1]) == 0xB5) {
      current += unit[i];
      current += unit[i + 1];
      ++i;
      continue;
    }
    char ch = unit[i];
    if (ch == '*' || ch == '/' || ch == '^' || ch == '(' || ch == ')') {
      if (!current.empty()) {
        tokens.push_back({TokenType::Value, current});
        current.clear();
      }
      TokenType t;
      switch (ch) {
      case '*':
        t = TokenType::Mul;
        break;
      case '/':
        t = TokenType::Div;
        break;
      case '^':
        t = TokenType::Pow;
        break;
      case '(':
        t = TokenType::LParen;
        break;
      case ')':
        t = TokenType::RParen;
        break;
      default:
        t = TokenType::Value;
        break;
      }
      tokens.push_back({t, std::string(1, ch)});
    } else if (std::isspace(byte) == 0) {
      current += ch;
    }
  }
  if (!current.empty()) {
    tokens.push_back({TokenType::Value, current});
  }
  return tokens;
}

// ---- Shunting-Yard (infix -> RPN) ----

static std::vector<Token> shunting_yard(const std::vector<Token> &tokens) {
  std::vector<Token> output;
  std::vector<Token> operators;
  for (const auto &token : tokens) {
    switch (token.type) {
    case TokenType::Value:
      output.push_back(token);
      break;
    case TokenType::Mul:
    case TokenType::Div:
    case TokenType::Pow: {
      while (!operators.empty()) {
        const auto &top = operators.back();
        if (token.precedence() <= top.precedence()) {
          output.push_back(operators.back());
          operators.pop_back();
        } else {
          break;
        }
      }
      operators.push_back(token);
      break;
    }
    case TokenType::LParen:
      operators.push_back(token);
      break;
    case TokenType::RParen: {
      while (!operators.empty() &&
             operators.back().type != TokenType::LParen) {
        output.push_back(operators.back());
        operators.pop_back();
      }
      if (operators.empty() || operators.back().type != TokenType::LParen) {
        throw std::invalid_argument(
            "unit expression has unbalanced parentheses");
      }
      operators.pop_back();
      break;
    }
    }
  }
  while (!operators.empty()) {
    if (operators.back().type == TokenType::LParen ||
        operators.back().type == TokenType::RParen) {
      throw std::invalid_argument(
          "unit expression has unbalanced parentheses");
    }
    output.push_back(operators.back());
    operators.pop_back();
  }
  return output;
}

// ---- AST evaluator ----

struct UnitExpr;
using UnitExprPtr = std::unique_ptr<UnitExpr>;

struct UnitExpr {
  struct Val {
    UnitValue value;
    std::string name;
  };
  struct Mul {
    UnitExprPtr lhs;
    UnitExprPtr rhs;
  };
  struct Div {
    UnitExprPtr lhs;
    UnitExprPtr rhs;
  };
  struct Pow {
    UnitExprPtr base;
    UnitExprPtr exponent;
  };

  std::variant<Val, Mul, Div, Pow> data;

  std::string to_string() const {
    return std::visit(
        [](const auto &v) -> std::string {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, Val>) {
            return v.name;
          } else if constexpr (std::is_same_v<T, Mul>) {
            return "(" + v.lhs->to_string() + " * " + v.rhs->to_string() + ")";
          } else if constexpr (std::is_same_v<T, Div>) {
            return "(" + v.lhs->to_string() + " / " + v.rhs->to_string() + ")";
          } else if constexpr (std::is_same_v<T, Pow>) {
            return "(" + v.base->to_string() + " ^ " +
                   v.exponent->to_string() + ")";
          }
        },
        data);
  }

  UnitValue eval() const {
    return std::visit(
        [](const auto &v) -> UnitValue {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, Val>) {
            return v.value;
          } else if constexpr (std::is_same_v<T, Mul>) {
            auto l = v.lhs->eval();
            auto r = v.rhs->eval();
            double result_factor = l.factor * r.factor;
            if (!std::isfinite(result_factor)) {
              throw std::invalid_argument(
                  "unit conversion overflows: multiplication of '" +
                  v.lhs->to_string() + " * " + v.rhs->to_string() + "'");
            }
            return {result_factor, l.dim * r.dim};
          } else if constexpr (std::is_same_v<T, Div>) {
            auto l = v.lhs->eval();
            auto r = v.rhs->eval();
            double result_factor = l.factor / r.factor;
            if (!std::isfinite(result_factor)) {
              throw std::invalid_argument(
                  "unit conversion overflows: division of '" +
                  v.lhs->to_string() + " / " + v.rhs->to_string() + "'");
            }
            return {result_factor, l.dim / r.dim};
          } else if constexpr (std::is_same_v<T, Pow>) {
            auto b = v.base->eval();
            auto e = v.exponent->eval();
            if (e.dim != DIM_NONE) {
              throw std::invalid_argument(
                  "exponent must be dimensionless, got dimension " +
                  e.dim.to_string() + " for '" + v.exponent->to_string() + "'");
            }
            double result_factor = std::pow(b.factor, e.factor);
            if (!std::isfinite(result_factor)) {
              throw std::invalid_argument(
                  "unit conversion overflows: exponentiation of '" +
                  v.base->to_string() + " ^ " + v.exponent->to_string() + "'");
            }
            return {result_factor, b.dim.pow(e.factor)};
          }
        },
        data);
  }
};

static UnitExprPtr read_expr(std::vector<Token> &stream) {
  if (stream.empty()) {
    throw std::invalid_argument(
        "malformed unit expression: missing a value");
  }
  auto token = stream.back();
  stream.pop_back();
  switch (token.type) {
  case TokenType::Value: {
    auto lower = to_lower(token.value);
    auto it = BASE_UNITS.find(lower);
    if (it != BASE_UNITS.end()) {
      auto expr = std::make_unique<UnitExpr>();
      expr->data = UnitExpr::Val{it->second, token.value};
      return expr;
    }
    try {
      double val = std::stod(token.value);
      auto expr = std::make_unique<UnitExpr>();
      expr->data = UnitExpr::Val{{val, DIM_NONE}, token.value};
      return expr;
    } catch (...) {
      throw std::invalid_argument("unknown unit '" + token.value + "'");
    }
  }
  case TokenType::Mul: {
    auto rhs = read_expr(stream);
    auto lhs = read_expr(stream);
    auto expr = std::make_unique<UnitExpr>();
    expr->data = UnitExpr::Mul{std::move(lhs), std::move(rhs)};
    return expr;
  }
  case TokenType::Div: {
    auto rhs = read_expr(stream);
    auto lhs = read_expr(stream);
    auto expr = std::make_unique<UnitExpr>();
    expr->data = UnitExpr::Div{std::move(lhs), std::move(rhs)};
    return expr;
  }
  case TokenType::Pow: {
    auto exponent = read_expr(stream);
    auto base = read_expr(stream);
    auto expr = std::make_unique<UnitExpr>();
    expr->data = UnitExpr::Pow{std::move(base), std::move(exponent)};
    return expr;
  }
  default:
    throw std::invalid_argument("unexpected symbol in unit expression: " +
                                token.as_str());
  }
}

static UnitValue parse_unit_expression(const std::string &unit) {
  if (unit.empty())
    return {1.0, DIM_NONE};
  auto tokens = tokenize(unit);
  if (tokens.empty())
    return {1.0, DIM_NONE};
  auto rpn = shunting_yard(tokens);
  auto ast = read_expr(rpn);
  if (!rpn.empty()) {
    std::string remaining;
    for (const auto &t : rpn) {
      if (!remaining.empty())
        remaining += " ";
      remaining += t.as_str();
    }
    throw std::invalid_argument("malformed unit expression: leftover input '" +
                                remaining + "'");
  }
  return ast->eval();
}

// ---- Quantity dimension map ----

static const auto QUANTITY_DIMS = std::unordered_map<std::string, Dimension>{
    {"length", DIM_LENGTH},
    {"energy", DIM_ENERGY},
    {"force", {{1, -2, 1, 0, 0}}},
    {"pressure", {{-1, -2, 1, 0, 0}}},
    {"momentum", {{1, -1, 1, 0, 0}}},
    {"mass", DIM_MASS},
    {"velocity", {{1, -1, 0, 0, 0}}},
    {"charge", DIM_CHARGE},
};

// ---- Public API ----

double unit_conversion_factor(const std::string &from_unit,
                              const std::string &to_unit) {
  if (from_unit.empty() || to_unit.empty())
    return 1.0;
  auto from = parse_unit_expression(from_unit);
  auto to = parse_unit_expression(to_unit);
  if (from.dim != to.dim) {
    throw std::invalid_argument("dimension mismatch: '" + from_unit +
                                "' has dimension " + from.dim.to_string() +
                                " but '" + to_unit + "' has dimension " +
                                to.dim.to_string());
  }
  return from.factor / to.factor;
}

void validate_unit(const std::string &quantity, const std::string &unit) {
  if (quantity.empty() || unit.empty())
    return;
  auto parsed = parse_unit_expression(unit);
  auto it = QUANTITY_DIMS.find(quantity);
  if (it != QUANTITY_DIMS.end()) {
    if (parsed.dim != it->second) {
      throw std::invalid_argument("unit '" + unit + "' has dimension " +
                                  parsed.dim.to_string() +
                                  " which is incompatible with quantity '" +
                                  quantity + "' (expected " +
                                  it->second.to_string() + ")");
    }
  }
}

} // namespace rgpot::units
