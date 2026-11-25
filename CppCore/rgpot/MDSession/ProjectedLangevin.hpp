#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file ProjectedLangevin.hpp
 * @brief BAOAB Langevin dynamics on a hyperplane, independent of the engine.
 *
 * Every MD session engine (OpenMM, GROMACS nblib, a LAMMPS plugin) hands this
 * driver the one thing it cannot supply itself: the potential energy and
 * forces at a set of positions. The driver owns the rest, so every engine
 * integrates the same constrained dynamics and draws the same random numbers
 * for the same seed.
 *
 * The dynamics is Langevin dynamics on the full 3N-dimensional configuration
 * space restricted to the hyperplane
 *
 *   sigma(x) = n . (x - p) = 0,
 *
 * with the unit normal n and the point p given in the flat atom-major
 * coordinates of the caller, and to any fixed distance constraints between
 * pairs of atoms. Masses enter through the metric M, so a constraint on
 * mass-weighted coordinates is the same hyperplane for every atom. One step
 * is
 *
 *   B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2)
 *
 * with the force kick B, the drift A and the Ornstein-Uhlenbeck update O.
 * The force of the first kick is evaluated at the constrained start, never at
 * the caller's raw positions. After every move that can leave the constraint
 * manifold, positions are corrected with
 * x -= M^-1 n sigma / (n M^-1 n) and velocities with
 * v -= M^-1 n (n . v) / (n M^-1 n). The velocity correction after a force
 * kick is the impulse of the constraint force lambda n, with
 *
 *   lambda = -(n . M^-1 F) / (n M^-1 n),
 *
 * and one lambda per step is reported: the mean of the values of the two
 * force kicks of the step, which is the constraint impulse divided by dt. For
 * a plane of fixed normal this is the Lagrange multiplier whose mean over the
 * constrained ensemble is the derivative of the free energy with respect to
 * the plane offset. With distance constraints present, the correction is
 * the projection on all constraint gradients J, v -= M^-1 J^T (J M^-1 J^T)^-1
 * J v, found with the conjugate gradient method, and lambda is the plane
 * entry of -(J M^-1 J^T)^-1 (J M^-1 F + Jdot v). The positions are corrected by the
 * SHAKE iteration with the gradients of the positions before the drift, and
 * the velocity takes the impulse of that correction (RATTLE), so without
 * friction the energy of a rigid group is conserved to second order.
 *
 * Planar hard walls n_w . (x - p_w) <= 0 bound the motion. A drift that ends
 * outside a wall is either reflected (the position is mirrored and the
 * velocity is reflected in the mass-weighted metric, repeatedly until the
 * state is inside every wall) or rejected (the drift is undone and the
 * velocity is reflected). Both keep the velocity on the constraint manifold.
 * A state that cannot be returned inside the walls is an error.
 *
 * The random stream is a SplitMix64 generator owned by the caller. A
 * SeededStream keeps it across calls, so a run split in chunks draws the
 * same numbers as one long run.
 *
 * Arrays have one entry per coordinate. The driver does not know units: kT
 * is in mass times velocity squared of the arrays, friction in inverse time
 * and time_step in time.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rgpot::md {

/**
 * @brief SplitMix64 generator with Box-Muller normals.
 *
 * The stream is a pure function of the seed on every platform, unlike the
 * standard distributions, so a seeded run replays exactly. The spare normal
 * of the Box-Muller pair is part of the state.
 */
class SplitMix64 {
public:
  explicit SplitMix64(std::uint64_t seed) noexcept : state_(seed) {}

  std::uint64_t nextU64() noexcept {
    state_ += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

  /// Uniform in the open interval (0, 1).
  double nextUniform() noexcept {
    const std::uint64_t bits = nextU64() >> 11;
    return (static_cast<double>(bits) + 0.5) * 0x1.0p-53;
  }

  double nextNormal() noexcept {
    if (hasSpare_) {
      hasSpare_ = false;
      return spare_;
    }
    const double u1 = nextUniform();
    const double u2 = nextUniform();
    const double radius = std::sqrt(-2.0 * std::log(u1));
    const double angle = 2.0 * kPi * u2;
    spare_ = radius * std::sin(angle);
    hasSpare_ = true;
    return radius * std::cos(angle);
  }

private:
  static constexpr double kPi = 3.14159265358979323846;
  std::uint64_t state_;
  bool hasSpare_ = false;
  double spare_ = 0.0;
};

/**
 * @brief The random stream of a session, identified by its seed.
 *
 * Asking for the seed of the live stream continues it, with the generator
 * state and the Box-Muller spare as the last call left them. Asking for a
 * different seed starts that stream from its beginning, with no spare. Two
 * sessions given the same seeds in the same order therefore draw identical
 * numbers, and a sampler that passes one run seed to every call gets one
 * uncorrelated stream. reset() ends the live stream, so that setting a new
 * state and running with the same seed replays the same noise.
 */
class SeededStream {
public:
  SplitMix64 &forSeed(std::uint64_t seed) {
    if (!active_ || seed != seed_) {
      rng_ = SplitMix64(seed);
      seed_ = seed;
      active_ = true;
    }
    return rng_;
  }

  void reset() noexcept { active_ = false; }

private:
  bool active_ = false;
  std::uint64_t seed_ = 0;
  SplitMix64 rng_{0};
};

/// What happens to a drift that ends outside a wall.
enum class WallMode { Reflect, Reject };

/// One planar wall n . (x - point) <= 0 over all coordinates.
struct WallSpec {
  std::vector<double> point;
  std::vector<double> normal;
};

/// The hyperplane n . (x - point) = 0.
struct PlaneSpec {
  std::vector<double> point;
  std::vector<double> normal;
};

/// A fixed distance between atoms first and second; atom a owns the
/// coordinates 3a, 3a + 1 and 3a + 2.
struct DistanceConstraint {
  std::size_t first = 0;
  std::size_t second = 0;
  double length = 0.0;
};

struct LangevinSettings {
  double kT = 0.0;
  double friction = 0.0;
  double timeStep = 0.0;
  std::uint64_t steps = 0;
  /// Seed of the stream used by the overload of run that has no stream.
  std::uint64_t seed = 0;
  WallMode wallMode = WallMode::Reflect;
  /// Record the constraint force of every step in LangevinResult.
  bool recordConstraintForce = false;
};

struct LangevinResult {
  std::uint64_t wallHits = 0;
  /// One entry per step when recorded: the Lagrange multiplier lambda of the
  /// plane, in force units of the caller's arrays. Empty without a plane.
  std::vector<double> constraintForce;
  /// One entry per step when recorded: Z = 1 / [(J M^-1 J^T)^-1]_plane at the
  /// end of the step, J the gradients of the plane and of the distances. The
  /// ensemble on the plane carries the factor Z^(1/2), so a free energy along
  /// the plane offset is that of the mean multiplier minus kT ln <Z^(-1/2)>.
  /// Without distances Z is the constant n M^-1 n.
  std::vector<double> constraintMetric;
  /// Forces at the final positions and the potential energy there.
  std::vector<double> forces;
  double potential = 0.0;
};

/// Potential energy at the positions; the forces are written to the second
/// argument. Both arrays have one entry per coordinate.
using ForceFunction = std::function<double(const double *, double *)>;

namespace detail {

inline void requireFiniteArray(const std::vector<double> &values,
                               const char *what) {
  for (const double value : values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(std::string(what) + " must be finite");
    }
  }
}

} // namespace detail

/**
 * @brief Maxwell-Boltzmann velocities, optionally on the plane n . v = 0.
 *
 * Each coordinate is drawn from a normal distribution of variance kT / m;
 * a coordinate of zero mass stays at rest. With a normal the draw is
 * conditioned on n . v = 0, which for this Gaussian is the projection
 * v -= M^-1 n (n . v) / (n M^-1 n).
 *
 * @param masses one mass per coordinate
 * @param kT in mass times velocity squared
 * @param normal null, or one entry per coordinate; need not be unit length
 */
inline std::vector<double> maxwellBoltzmannVelocities(
    const std::vector<double> &masses, double kT, std::uint64_t seed,
    const double *normal) {
  if (!(kT >= 0.0) || !std::isfinite(kT)) {
    throw std::invalid_argument("kT must be finite and not negative");
  }
  const std::size_t dim = masses.size();
  std::vector<double> velocity(dim, 0.0);
  SplitMix64 rng(seed);
  for (std::size_t k = 0; k < dim; ++k) {
    if (masses[k] > 0.0) {
      velocity[k] = std::sqrt(kT / masses[k]) * rng.nextNormal();
    }
  }
  if (normal != nullptr) {
    double along = 0.0;
    double weighted = 0.0;
    for (std::size_t k = 0; k < dim; ++k) {
      if (masses[k] > 0.0) {
        along += normal[k] * velocity[k];
        weighted += normal[k] * normal[k] / masses[k];
      }
    }
    if (!(weighted > 0.0) || !std::isfinite(weighted)) {
      throw std::invalid_argument(
          "the normal must be finite and act on a massive coordinate");
    }
    const double scale = along / weighted;
    for (std::size_t k = 0; k < dim; ++k) {
      if (masses[k] > 0.0) {
        velocity[k] -= scale * normal[k] / masses[k];
      }
    }
  }
  return velocity;
}

/**
 * @brief The projected BAOAB integrator.
 *
 * Construct it with the mass of every coordinate and the fixed distances of
 * the system, then call run. The object holds no state between runs.
 */
class ProjectedLangevin {
public:
  /// masses: one value per coordinate; zero marks a coordinate that does not
  /// move. distances: constraints that hold in every run.
  explicit ProjectedLangevin(std::vector<double> masses,
                             std::vector<DistanceConstraint> distances = {})
      : invMass_(masses.size(), 0.0), dim_(masses.size()),
        distances_(std::move(distances)) {
    bool massive = false;
    for (std::size_t k = 0; k < dim_; ++k) {
      if (!(masses[k] >= 0.0) || !std::isfinite(masses[k])) {
        throw std::invalid_argument("masses must be finite and not negative");
      }
      if (masses[k] > 0.0) {
        invMass_[k] = 1.0 / masses[k];
        massive = true;
      }
    }
    if (!massive) {
      throw std::invalid_argument("at least one coordinate needs a mass");
    }
    if (dim_ % 3 != 0 && !distances_.empty()) {
      throw std::invalid_argument(
          "distance constraints need three coordinates per atom");
    }
    for (const DistanceConstraint &c : distances_) {
      if (3 * c.first + 3 > dim_ || 3 * c.second + 3 > dim_ ||
          c.first == c.second || !(c.length > 0.0) ||
          !std::isfinite(c.length)) {
        throw std::invalid_argument(
            "a distance constraint needs two distinct atoms and a positive "
            "length");
      }
    }
  }

  [[nodiscard]] std::size_t dimension() const { return dim_; }

  /**
   * Integrate settings.steps steps from the positions x and velocities v,
   * which are updated in place, drawing noise from rng. forces is called once
   * before the first step and once per step. The state is first moved onto
   * the constraints. Zero steps change nothing.
   */
  LangevinResult run(const LangevinSettings &settings, const PlaneSpec *plane,
                     const std::vector<WallSpec> &walls, double *x, double *v,
                     const ForceFunction &forces, SplitMix64 &rng) const {
    validate(settings);
    LangevinResult result;
    if (settings.steps == 0) {
      return result;
    }
    PreparedPlane prepared;
    const bool hasPlane = plane != nullptr;
    if (hasPlane) {
      prepared = preparePlane(*plane);
    }
    const bool constrained = hasPlane || !distances_.empty();
    const bool general = !distances_.empty();
    Context ctx{prepared, hasPlane, general};
    const std::vector<PreparedWall> prepWalls = prepareWalls(walls, prepared);
    if (constrained) {
      std::vector<double> reference(x, x + dim_);
      projectPosition(ctx, reference.data(), x);
      projectVelocity(ctx, x, v);
    }
    for (const PreparedWall &wall : prepWalls) {
      const Frame frame = frameOf(ctx, wall, x);
      if (frame.scale2 > 0.0 && wall.value(x) > kInsideTolerance) {
        throw std::invalid_argument("the positions start outside a wall");
      }
    }

    const double half = 0.5 * settings.timeStep;
    const double decay = std::exp(-settings.friction * settings.timeStep);
    const double spread = -std::expm1(-2.0 * settings.friction *
                                      settings.timeStep);
    std::vector<double> force(dim_, 0.0);
    std::vector<double> previous(dim_, 0.0);
    std::vector<double> drifted(general ? dim_ : 0, 0.0);
    result.potential = forces(x, force.data());
    double lambdaStart = hasPlane ? constraintForce(ctx, x, v, force) : 0.0;
    if (hasPlane && settings.recordConstraintForce) {
      result.constraintForce.reserve(settings.steps);
      result.constraintMetric.reserve(settings.steps);
    }

    for (std::uint64_t step = 0; step < settings.steps; ++step) {
      kick(half, force, v);
      if (constrained) {
        projectVelocity(ctx, x, v);
      }

      for (int pass = 0; pass < 2; ++pass) {
        std::copy(x, x + dim_, previous.begin());
        for (std::size_t k = 0; k < dim_; ++k) {
          x[k] += half * v[k];
        }
        if (constrained) {
          if (general) {
            std::copy(x, x + dim_, drifted.begin());
          }
          projectPosition(ctx, previous.data(), x);
          if (general) {
            // The constraint impulse of the drift. A rigid group turns during
            // the drift, and projecting v onto the new tangent space alone
            // would shorten it, a loss of kinetic energy on every step.
            for (std::size_t k = 0; k < dim_; ++k) {
              v[k] += (x[k] - drifted[k]) / half;
            }
            projectVelocity(ctx, x, v);
          }
        }
        result.wallHits += applyWalls(ctx, prepWalls, settings.wallMode,
                                      previous, x, v);
        if (pass == 0) {
          thermalize(settings, decay, spread, rng, v);
          if (constrained) {
            projectVelocity(ctx, x, v);
          }
        }
      }

      result.potential = forces(x, force.data());
      kick(half, force, v);
      if (constrained) {
        projectVelocity(ctx, x, v);
      }
      if (hasPlane) {
        const double lambdaEnd = constraintForce(ctx, x, v, force);
        if (settings.recordConstraintForce) {
          result.constraintForce.push_back(0.5 * (lambdaStart + lambdaEnd));
          result.constraintMetric.push_back(constraintMetric(ctx, x));
        }
        lambdaStart = lambdaEnd;
      }
    }
    result.forces = std::move(force);
    return result;
  }

  /// run with a stream started from settings.seed.
  LangevinResult run(const LangevinSettings &settings, const PlaneSpec *plane,
                     const std::vector<WallSpec> &walls, double *x, double *v,
                     const ForceFunction &forces) const {
    SplitMix64 rng(settings.seed);
    return run(settings, plane, walls, x, v, forces, rng);
  }

private:
  struct PreparedPlane {
    std::vector<double> point;
    std::vector<double> normal;
    std::vector<double> direction; //!< M^-1 n / (n M^-1 n)
    double weighted = 0.0;         //!< n M^-1 n
  };

  struct PreparedWall {
    std::vector<double> point;
    std::vector<double> normal;
    std::vector<double> tangent; //!< the normal's part tangent to the plane
    std::vector<double> mapped;  //!< M^-1 tangent
    double scale2 = 0.0;         //!< tangent M^-1 tangent
    double full2 = 0.0;          //!< normal M^-1 normal

    [[nodiscard]] double value(const double *x) const {
      double sum = 0.0;
      for (std::size_t k = 0; k < normal.size(); ++k) {
        sum += normal[k] * (x[k] - point[k]);
      }
      return sum;
    }
  };

  /// The part of a wall normal tangent to the constraints at a position.
  struct Frame {
    std::vector<double> tangent;
    std::vector<double> mapped;
    double scale2 = 0.0;
  };

  struct Context {
    const PreparedPlane &plane;
    bool hasPlane;
    bool general; //!< distance constraints present
  };

  /// The constraint gradients at one position: the plane normal, then one
  /// row per distance constraint.
  struct Gradients {
    const double *planeNormal = nullptr;
    struct Pair {
      std::size_t first;
      std::size_t second;
      double delta[3];
    };
    std::vector<Pair> pairs;

    [[nodiscard]] std::size_t rows() const {
      return (planeNormal != nullptr ? 1 : 0) + pairs.size();
    }
  };

  static constexpr double kInsideTolerance = 1.0e-9;
  static constexpr int kMaxWallPasses = 100000;

  void validate(const LangevinSettings &settings) const {
    if (!(settings.kT > 0.0) || !std::isfinite(settings.kT)) {
      throw std::invalid_argument("kT must be finite and positive");
    }
    if (!(settings.friction >= 0.0) || !std::isfinite(settings.friction)) {
      throw std::invalid_argument(
          "friction must be finite and not negative");
    }
    if (!(settings.timeStep > 0.0) || !std::isfinite(settings.timeStep)) {
      throw std::invalid_argument("time_step must be finite and positive");
    }
  }

  [[nodiscard]] PreparedPlane preparePlane(const PlaneSpec &plane) const {
    if (plane.point.size() != dim_ || plane.normal.size() != dim_) {
      throw std::invalid_argument(
          "the plane point and normal need one entry per coordinate");
    }
    detail::requireFiniteArray(plane.point, "the plane point");
    detail::requireFiniteArray(plane.normal, "the plane normal");
    double norm2 = 0.0;
    for (const double value : plane.normal) {
      norm2 += value * value;
    }
    if (!(norm2 > 0.0) || !std::isfinite(norm2)) {
      throw std::invalid_argument("the plane normal must not be zero");
    }
    PreparedPlane out;
    out.point = plane.point;
    const double inverse = 1.0 / std::sqrt(norm2);
    out.normal.resize(dim_);
    for (std::size_t k = 0; k < dim_; ++k) {
      out.normal[k] = plane.normal[k] * inverse;
      out.weighted += out.normal[k] * out.normal[k] * invMass_[k];
    }
    if (!(out.weighted > 0.0) || !std::isfinite(out.weighted)) {
      throw std::invalid_argument(
          "the plane normal must act on a massive coordinate");
    }
    out.direction.resize(dim_);
    for (std::size_t k = 0; k < dim_; ++k) {
      out.direction[k] = out.normal[k] * invMass_[k] / out.weighted;
    }
    return out;
  }

  [[nodiscard]] std::vector<PreparedWall>
  prepareWalls(const std::vector<WallSpec> &walls,
               const PreparedPlane &plane) const {
    std::vector<PreparedWall> out;
    out.reserve(walls.size());
    const bool constrained = !plane.normal.empty();
    for (const WallSpec &wall : walls) {
      if (wall.point.size() != dim_ || wall.normal.size() != dim_) {
        throw std::invalid_argument(
            "a wall point and normal need one entry per coordinate");
      }
      detail::requireFiniteArray(wall.point, "a wall point");
      detail::requireFiniteArray(wall.normal, "a wall normal");
      double norm2 = 0.0;
      for (const double value : wall.normal) {
        norm2 += value * value;
      }
      if (!(norm2 > 0.0) || !std::isfinite(norm2)) {
        throw std::invalid_argument("a wall normal must not be zero");
      }
      PreparedWall prepared;
      prepared.point = wall.point;
      prepared.normal.resize(dim_);
      const double inverse = 1.0 / std::sqrt(norm2);
      double alongPlane = 0.0;
      for (std::size_t k = 0; k < dim_; ++k) {
        prepared.normal[k] = wall.normal[k] * inverse;
        if (constrained) {
          alongPlane += plane.normal[k] * invMass_[k] * prepared.normal[k];
        }
        prepared.full2 +=
            prepared.normal[k] * prepared.normal[k] * invMass_[k];
      }
      prepared.tangent = prepared.normal;
      if (constrained) {
        const double coefficient = alongPlane / plane.weighted;
        for (std::size_t k = 0; k < dim_; ++k) {
          prepared.tangent[k] -= coefficient * plane.normal[k];
        }
      }
      prepared.mapped.resize(dim_);
      for (std::size_t k = 0; k < dim_; ++k) {
        prepared.mapped[k] = prepared.tangent[k] * invMass_[k];
        prepared.scale2 += prepared.tangent[k] * prepared.mapped[k];
      }
      // A wall whose normal lies along the constraint cannot be reached on
      // the plane; rounding leaves a residue far below the wall's own scale.
      if (!(prepared.scale2 > 1.0e-20 * prepared.full2)) {
        prepared.scale2 = 0.0;
      }
      out.push_back(std::move(prepared));
    }
    return out;
  }

  void kick(double half, const std::vector<double> &force, double *v) const {
    for (std::size_t k = 0; k < dim_; ++k) {
      v[k] += half * force[k] * invMass_[k];
    }
  }

  void thermalize(const LangevinSettings &settings, double decay,
                  double spread, SplitMix64 &rng, double *v) const {
    if (settings.friction == 0.0) {
      return;
    }
    const double amplitude = std::sqrt(settings.kT * spread);
    for (std::size_t k = 0; k < dim_; ++k) {
      if (invMass_[k] > 0.0) {
        v[k] = decay * v[k] + amplitude * std::sqrt(invMass_[k]) *
                                  rng.nextNormal();
      }
    }
  }

  // -- The plane alone: closed forms --------------------------------------

  static void projectPlanePosition(const PreparedPlane &plane, double *x) {
    double sigma = 0.0;
    for (std::size_t k = 0; k < plane.normal.size(); ++k) {
      sigma += plane.normal[k] * (x[k] - plane.point[k]);
    }
    for (std::size_t k = 0; k < plane.normal.size(); ++k) {
      x[k] -= plane.direction[k] * sigma;
    }
  }

  static void projectPlaneVelocity(const PreparedPlane &plane, double *v) {
    double along = 0.0;
    for (std::size_t k = 0; k < plane.normal.size(); ++k) {
      along += plane.normal[k] * v[k];
    }
    for (std::size_t k = 0; k < plane.normal.size(); ++k) {
      v[k] -= plane.direction[k] * along;
    }
  }

  // -- Distance constraints and the plane together ------------------------

  [[nodiscard]] Gradients gradients(const Context &ctx, const double *x) const {
    Gradients out;
    if (ctx.hasPlane) {
      out.planeNormal = ctx.plane.normal.data();
    }
    out.pairs.reserve(distances_.size());
    for (const DistanceConstraint &c : distances_) {
      Gradients::Pair pair{c.first, c.second, {0.0, 0.0, 0.0}};
      for (std::size_t d = 0; d < 3; ++d) {
        pair.delta[d] = x[3 * c.first + d] - x[3 * c.second + d];
      }
      out.pairs.push_back(pair);
    }
    return out;
  }

  /// rows = J w
  static void applyJ(const Gradients &g, const double *w, std::size_t dim,
                     std::vector<double> &rows) {
    rows.assign(g.rows(), 0.0);
    std::size_t row = 0;
    if (g.planeNormal != nullptr) {
      double sum = 0.0;
      for (std::size_t k = 0; k < dim; ++k) {
        sum += g.planeNormal[k] * w[k];
      }
      rows[row++] = sum;
    }
    for (const Gradients::Pair &pair : g.pairs) {
      double sum = 0.0;
      for (std::size_t d = 0; d < 3; ++d) {
        sum += pair.delta[d] * (w[3 * pair.first + d] - w[3 * pair.second + d]);
      }
      rows[row++] = sum;
    }
  }

  /// out = J^T y
  void applyJt(const Gradients &g, const std::vector<double> &y,
               std::vector<double> &out) const {
    out.assign(dim_, 0.0);
    std::size_t row = 0;
    if (g.planeNormal != nullptr) {
      const double coefficient = y[row++];
      for (std::size_t k = 0; k < dim_; ++k) {
        out[k] += coefficient * g.planeNormal[k];
      }
    }
    for (const Gradients::Pair &pair : g.pairs) {
      const double coefficient = y[row++];
      for (std::size_t d = 0; d < 3; ++d) {
        out[3 * pair.first + d] += coefficient * pair.delta[d];
        out[3 * pair.second + d] -= coefficient * pair.delta[d];
      }
    }
  }

  /// y = (J M^-1 J^T) z
  void applyGram(const Gradients &g, const std::vector<double> &z,
                 std::vector<double> &y, std::vector<double> &work) const {
    applyJt(g, z, work);
    for (std::size_t k = 0; k < dim_; ++k) {
      work[k] *= invMass_[k];
    }
    applyJ(g, work.data(), dim_, y);
  }

  /// Solve (J M^-1 J^T) y = b by conjugate gradients.
  [[nodiscard]] std::vector<double> solveGram(const Gradients &g,
                                              const std::vector<double> &b)
      const {
    const std::size_t n = b.size();
    std::vector<double> y(n, 0.0);
    std::vector<double> r = b;
    std::vector<double> p = b;
    std::vector<double> ap(n, 0.0);
    std::vector<double> work;
    double rr = 0.0;
    double bb = 0.0;
    for (const double value : b) {
      rr += value * value;
    }
    bb = rr;
    if (bb == 0.0) {
      return y;
    }
    const std::size_t limit = 4 * n + 40;
    for (std::size_t iteration = 0;
         iteration < limit && rr > 1.0e-30 * bb; ++iteration) {
      applyGram(g, p, ap, work);
      double curvature = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        curvature += p[i] * ap[i];
      }
      if (!(curvature > 0.0) || !std::isfinite(curvature)) {
        throw std::runtime_error(
            "the constraints are degenerate: their gradients are dependent "
            "or act only on massless coordinates");
      }
      const double alpha = rr / curvature;
      double next = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        y[i] += alpha * p[i];
        r[i] -= alpha * ap[i];
        next += r[i] * r[i];
      }
      const double beta = next / rr;
      rr = next;
      for (std::size_t i = 0; i < n; ++i) {
        p[i] = r[i] + beta * p[i];
      }
    }
    if (!(rr <= 1.0e-16 * bb)) {
      throw std::runtime_error("the constraint solve did not converge");
    }
    return y;
  }

  /// Constraint values: the plane offset, then (|d|^2 - length^2) / 2.
  void constraintValues(const Context &ctx, const double *x,
                        std::vector<double> &values) const {
    values.clear();
    if (ctx.hasPlane) {
      double sigma = 0.0;
      for (std::size_t k = 0; k < dim_; ++k) {
        sigma += ctx.plane.normal[k] * (x[k] - ctx.plane.point[k]);
      }
      values.push_back(sigma);
    }
    for (const DistanceConstraint &c : distances_) {
      double sum = 0.0;
      for (std::size_t d = 0; d < 3; ++d) {
        const double delta = x[3 * c.first + d] - x[3 * c.second + d];
        sum += delta * delta;
      }
      values.push_back(0.5 * (sum - c.length * c.length));
    }
  }

  /// Move x onto the constraints, with the gradients taken at reference.
  /// reference and x may not alias.
  void projectPosition(const Context &ctx, const double *reference,
                       double *x) const {
    if (!ctx.general) {
      projectPlanePosition(ctx.plane, x);
      return;
    }
    const Gradients g = gradients(ctx, reference);
    // The plane tolerance is relative to the coordinates, with a floor of
    // one length unit so a geometry at the origin can still converge.
    double scale = 1.0;
    for (std::size_t k = 0; k < dim_; ++k) {
      scale = std::max(scale, std::abs(x[k]));
      if (ctx.hasPlane) {
        scale = std::max(scale, std::abs(ctx.plane.point[k]));
      }
    }
    std::vector<double> values;
    std::vector<double> correction;
    constexpr int kMaxIterations = 200;
    for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
      constraintValues(ctx, x, values);
      bool done = true;
      std::size_t row = 0;
      if (ctx.hasPlane) {
        done = std::abs(values[row++]) <= 1.0e-13 * scale;
      }
      for (const DistanceConstraint &c : distances_) {
        done = done &&
               std::abs(values[row++]) <= 1.0e-13 * c.length * c.length;
      }
      if (done) {
        return;
      }
      const std::vector<double> multipliers = solveGram(g, values);
      applyJt(g, multipliers, correction);
      for (std::size_t k = 0; k < dim_; ++k) {
        x[k] -= correction[k] * invMass_[k];
      }
    }
    throw std::runtime_error("the position constraints did not converge");
  }

  /// Remove from v the part along the constraint gradients at x.
  void projectVelocity(const Context &ctx, const double *x,
                       double *v) const {
    if (!ctx.general) {
      projectPlaneVelocity(ctx.plane, v);
      return;
    }
    const Gradients g = gradients(ctx, x);
    std::vector<double> rows;
    applyJ(g, v, dim_, rows);
    const std::vector<double> multipliers = solveGram(g, rows);
    std::vector<double> correction;
    applyJt(g, multipliers, correction);
    for (std::size_t k = 0; k < dim_; ++k) {
      v[k] -= correction[k] * invMass_[k];
    }
  }

  /// lambda of the plane for the force at x and the velocity v on the
  /// constraints. A distance is curved, so holding it needs the centripetal
  /// force of v, and through the plane that force enters lambda.
  [[nodiscard]] double constraintForce(const Context &ctx, const double *x,
                                       const double *v,
                                       const std::vector<double> &force) const {
    if (!ctx.general) {
      double along = 0.0;
      for (std::size_t k = 0; k < dim_; ++k) {
        along += ctx.plane.normal[k] * invMass_[k] * force[k];
      }
      return -along / ctx.plane.weighted;
    }
    const Gradients g = gradients(ctx, x);
    std::vector<double> mapped(dim_);
    for (std::size_t k = 0; k < dim_; ++k) {
      mapped[k] = force[k] * invMass_[k];
    }
    std::vector<double> rows;
    applyJ(g, mapped.data(), dim_, rows);
    std::size_t row = g.planeNormal != nullptr ? 1 : 0;
    for (const Gradients::Pair &pair : g.pairs) {
      for (std::size_t d = 0; d < 3; ++d) {
        const double relative = v[3 * pair.first + d] - v[3 * pair.second + d];
        rows[row] += relative * relative;
      }
      ++row;
    }
    for (double &value : rows) {
      value = -value;
    }
    return solveGram(g, rows).front();
  }

  /// Z = 1 / [(J M^-1 J^T)^-1]_plane at x.
  [[nodiscard]] double constraintMetric(const Context &ctx,
                                        const double *x) const {
    if (!ctx.general) {
      return ctx.plane.weighted;
    }
    const Gradients g = gradients(ctx, x);
    std::vector<double> unit(g.rows(), 0.0);
    unit.front() = 1.0;
    return 1.0 / solveGram(g, unit).front();
  }

  [[nodiscard]] Frame frameOf(const Context &ctx, const PreparedWall &wall,
                              const double *x) const {
    Frame frame;
    if (!ctx.general) {
      frame.tangent = wall.tangent;
      frame.mapped = wall.mapped;
      frame.scale2 = wall.scale2;
      return frame;
    }
    const Gradients g = gradients(ctx, x);
    std::vector<double> mapped(dim_);
    for (std::size_t k = 0; k < dim_; ++k) {
      mapped[k] = wall.normal[k] * invMass_[k];
    }
    std::vector<double> rows;
    applyJ(g, mapped.data(), dim_, rows);
    const std::vector<double> multipliers = solveGram(g, rows);
    std::vector<double> along;
    applyJt(g, multipliers, along);
    frame.tangent.resize(dim_);
    frame.mapped.resize(dim_);
    for (std::size_t k = 0; k < dim_; ++k) {
      frame.tangent[k] = wall.normal[k] - along[k];
      frame.mapped[k] = frame.tangent[k] * invMass_[k];
      frame.scale2 += frame.tangent[k] * frame.mapped[k];
    }
    if (!(frame.scale2 > 1.0e-20 * wall.full2)) {
      frame.scale2 = 0.0;
    }
    return frame;
  }

  /// Reflect the velocity in the wall when it points outward.
  static void reflectVelocity(const Frame &frame, double *v) {
    double along = 0.0;
    for (std::size_t k = 0; k < frame.tangent.size(); ++k) {
      along += frame.tangent[k] * v[k];
    }
    if (along > 0.0) {
      const double scale = 2.0 * along / frame.scale2;
      for (std::size_t k = 0; k < frame.tangent.size(); ++k) {
        v[k] -= scale * frame.mapped[k];
      }
    }
  }

  std::uint64_t applyWalls(const Context &ctx,
                           const std::vector<PreparedWall> &walls,
                           WallMode mode, const std::vector<double> &previous,
                           double *x, double *v) const {
    if (walls.empty()) {
      return 0;
    }
    std::uint64_t hits = 0;
    const bool constrained = ctx.hasPlane || ctx.general;
    if (mode == WallMode::Reject) {
      bool outside = false;
      for (const PreparedWall &wall : walls) {
        if (wall.value(x) > 0.0) {
          const Frame frame = frameOf(ctx, wall, previous.data());
          if (frame.scale2 > 0.0) {
            reflectVelocity(frame, v);
            ++hits;
            outside = true;
          }
        }
      }
      if (outside) {
        std::copy(previous.begin(), previous.end(), x);
      }
      return hits;
    }
    std::vector<double> before(dim_);
    for (int pass = 0; pass < kMaxWallPasses; ++pass) {
      bool any = false;
      for (const PreparedWall &wall : walls) {
        const double excess = wall.value(x);
        if (!(excess > 0.0)) {
          continue;
        }
        const Frame frame = frameOf(ctx, wall, x);
        if (!(frame.scale2 > 0.0)) {
          continue;
        }
        std::copy(x, x + dim_, before.begin());
        const double scale = 2.0 * excess / frame.scale2;
        for (std::size_t k = 0; k < dim_; ++k) {
          x[k] -= scale * frame.mapped[k];
        }
        if (constrained) {
          projectPosition(ctx, before.data(), x);
        }
        reflectVelocity(frame, v);
        if (constrained) {
          projectVelocity(ctx, x, v);
        }
        ++hits;
        any = true;
      }
      if (!any) {
        return hits;
      }
    }
    throw std::runtime_error(
        "the walls could not return the state inside after "
        "repeated reflections");
  }

  std::vector<double> invMass_;
  std::size_t dim_;
  std::vector<DistanceConstraint> distances_;
};

} // namespace rgpot::md
