# Copyright 2023--present rgconf developers
@0xd782940f9a21b301;

# rgconf common primitives, typed field annotations, union-based presence wrappers,
# compatibility profiles, physical unit system, finite-difference stencils,
# periodic cell geometry, and neighbor-list configuration.

struct SchemaVersion {
  major @0 :UInt16 = 1;
  minor @1 :UInt16 = 1;
  patch @2 :UInt16 = 0;
}

# Named compatibility profiles freezing default policies across consumers.
enum CompatibilityProfile {
  rgconfCanonical @0;   # System-size-independent maxAtomL2 across optimizer, band, and saddle
  eonLegacyIni @1;      # Preserves legacy eOn [Optimizer] convergence_metric="norm" (l2Euclidean) while band/saddle use maxAtomL2
  eonServerYaml @2;     # Preserves eon-akmc server config.yaml defaults (job=akmc, ext_pot_path="ext_pot")
  gpridgePaper @3;      # Materializes modelProfile=gprdZbl, band.forceTolerance=0.0514221 eV/A, imageCount=nimages+2 (10)
  gpridgeLegacyToml @4; # Preserves legacy gpridge modelProfile=extended, band.forceTolerance=0.05 eV/A
  aseDropin @5;         # Preserves ASE fmax (maxAtomL2) and maxstep=0.2 Angstrom conventions
}

# Physical quantity dimensions for compile-time and load-time unit checking.
enum DimensionKind {
  dimensionless @0;
  length @1;
  energy @2;
  time @3;
  force @4;
  forceConstant @5;
  hessian @6;
  angle @7;
  mass @8;
  temperature @9;
  charge @10;
  pressureStress @11;
  frequencyWavenumber @12;
  dipole @13;
  quadrupole @14;
  polarizability @15;
  entropyHeatCapacity @16;
  frictionRate @17;
  memorySize @18;
  # Norm of a force in mass-weighted coordinates (LI-NEB opt/perp tolerances).
  massWeightedForce @19;
  inverseRootEnergy @20;
}

enum CanonicalUnit {
  none @0;
  angstrom @1;
  bohr @2;
  nanometer @3;
  ev @4;
  hartree @5;
  rydberg @6;
  kjPerMol @7;
  kcalPerMol @8;
  femtosecond @9;
  picosecond @10;
  second @11;
  eonTimeUnit @12; # 1 eOn time unit = 10.1805055 fs (sqrt(amu) * Angstrom / sqrt(eV))
  atomicTimeUnit @13;
  evPerAngstrom @14;
  hartreePerBohr @15;
  evPerAngstrom2 @16;
  hartreePerBohr2 @17;
  evPerAngstrom3 @18;
  hartreePerBohr3 @19;
  gpa @20;
  radian @21;
  degree @22;
  amu @23;
  electronMass @24;
  kelvin @25;
  elementaryCharge @26;
  invCm @27;
  debye @28;
  atomicDipoleAu @29;
  atomicQuadrupoleAu @30;
  atomicPolarizabilityAu @31;
  evPerKelvin @32;
  invFemtosecond @33;
  invPicosecond @34;
  invSecond @35;
  megabyte @36;
  gigabyte @37;
  # Mass-weighted force: √eV when lengths are absorbed into q = √m x (LI-NEB).
  sqrtEv @38;
  invSqrtEv @39;
}

enum StressSignConvention {
  cauchyPositiveTensile @0;      # sigma = +(1/V) dE/d(strain); ASE and solid-state NEB convention
  virialPositiveCompressive @1;  # W = -dE/d(strain) = -V * sigma; i-PI and OpenCPMD convention
}

struct UnitSystem {
  length @0 :CanonicalUnit = angstrom;
  energy @1 :CanonicalUnit = ev;
  time @2 :CanonicalUnit = femtosecond;
  angle @3 :CanonicalUnit = radian;
  mass @4 :CanonicalUnit = amu;
  temperature @5 :CanonicalUnit = kelvin;
  charge @6 :CanonicalUnit = elementaryCharge;
  pressureStress @7 :CanonicalUnit = evPerAngstrom3;
  stressSign @8 :StressSignConvention = cauchyPositiveTensile;
  frequency @9 :CanonicalUnit = invCm;
  dipole @10 :CanonicalUnit = debye;
  hessian @11 :CanonicalUnit = evPerAngstrom2;
}

# Per-field runtime mutability classification.
enum MutabilityTier {
  mutableLive @0;           # Safe to change mid-run without resetting solver state (e.g. forceTolerance, maxIterations, maxAtomMove)
  betweenStepsWithHook @1;  # Safe to change at step boundaries; triggers a deterministic transition hook (e.g. L-BFGS memory/method resets secant history; spring constant recomputes tangents)
  restartOnly @2;           # Structural/topological setting; requires checkpoint restart (e.g. atom count, fixed-atom mask, potential backend, manifold dimension)
}

enum DefaultKind {
  literalWireDefault @0; # Fixed literal default on the Cap'n Proto field
  profileDependent @1;   # Materialized from CompatibilityProfile when unset
  resolverDerived @2;    # Computed from another resolved field when unset (never overrides an explicit value)
}

struct FieldBounds {
  hasMinInclusive @0 :Bool = false;
  minInclusive @1 :Float64 = 0.0;
  hasMinExclusive @2 :Bool = false;
  minExclusive @3 :Float64 = 0.0;
  hasMaxInclusive @4 :Bool = false;
  maxInclusive @5 :Float64 = 0.0;
  hasMaxExclusive @6 :Bool = false;
  maxExclusive @7 :Float64 = 0.0;
  requireFinite @8 :Bool = true;
}

struct SourceAlias {
  format @0 :Text;             # "ini", "toml", "json", "yaml"
  section @1 :Text;            # e.g. "Nudged Elastic Band" or "band"
  key @2 :Text;                # e.g. "ew_ksp_min" or "nimages"
  transform @3 :Text = "";     # e.g. "total_from_intermediate_plus_2"
  deprecated @4 :Bool = false;
  deprecationNote @5 :Text = "";
}

struct FieldMeta {
  dimension @0 :DimensionKind = dimensionless;
  unit @1 :CanonicalUnit = none;
  bounds @2 :FieldBounds;
  aliases @3 :List(SourceAlias);
  cliFlags @4 :List(Text);
  applicability @5 :Text = ""; # Predicate over resolved RgConf, e.g. "optimizer.method == cg"
  defaultKind @6 :DefaultKind = literalWireDefault;
  resolverFunction @7 :Text = "";
  deprecated @8 :Bool = false;
  deprecationReason @9 :Text = "";
  # Loader-visible documentation. Empty means the preceding schema comment.
  doc @10 :Text = "";
}

annotation meta(field, struct, enum, enumerant) :FieldMeta;
annotation mutability(field) :MutabilityTier;

# Union-based presence wrappers for patch/overlay structs.
# Using `union { unset @0 :Void; value @1 :T; }` makes the invalid state
# `isSet = false, value = 90` unrepresentable on the wire.
struct OptFloat64 {
  union {
    unset @0 :Void;
    value @1 :Float64;
  }
}

struct OptInt64 {
  union {
    unset @0 :Void;
    value @1 :Int64;
  }
}

struct OptUInt64 {
  union {
    unset @0 :Void;
    value @1 :UInt64;
  }
}

struct OptInt32 {
  union {
    unset @0 :Void;
    value @1 :Int32;
  }
}

struct OptUInt32 {
  union {
    unset @0 :Void;
    value @1 :UInt32;
  }
}

struct OptUInt16 {
  union {
    unset @0 :Void;
    value @1 :UInt16;
  }
}

struct OptUInt8 {
  union {
    unset @0 :Void;
    value @1 :UInt8;
  }
}

struct OptBool {
  union {
    unset @0 :Void;
    value @1 :Bool;
  }
}

struct OptText {
  union {
    unset @0 :Void;
    value @1 :Text;
  }
}

struct OptInt32List {
  union {
    unset @0 :Void;
    value @1 :List(Int32);
  }
}

struct OptUInt32List {
  union {
    unset @0 :Void;
    value @1 :List(UInt32);
  }
}

struct OptFloat64List {
  union {
    unset @0 :Void;
    value @1 :List(Float64);
  }
}

struct OptTextList {
  union {
    unset @0 :Void;
    value @1 :List(Text);
  }
}

# Replaces signed/zero sentinels for iteration or step limits.
struct CountLimit {
  union {
    conditionDriven @0 :Void;
    limited @1 :UInt32;
  }
}

# Replaces negative sentinels such as rffFeatures = -1 ("auto") vs 0 ("disabled").
struct FeatureCount {
  union {
    disabled @0 :Void;
    autoSelect @1 :Void;
    count @2 :UInt32;
  }
}

enum FiniteDifferenceScheme {
  forwardOneSided @0;  # 1 perturbed force evaluation: (F(x + dr * v) - F(x)) / dr
  centralTwoPoint @1;  # 2 perturbed force evaluations: (F(x + dr * v) - F(x - dr * v)) / (2 * dr)
  centralFourPoint @2; # 4 perturbed force evaluations: 4th-order Richardson stencil
}

struct OptFiniteDifferenceScheme {
  union {
    unset @0 :Void;
    value @1 :FiniteDifferenceScheme;
  }
}

struct FiniteDifferenceConfig {
  stepAngstrom @0 :Float64 = 0.01 $mutability(mutableLive) $meta(dimension = length, unit = angstrom, bounds = (hasMinExclusive = true, minExclusive = 0.0), cliFlags = ["--fd-step"]);
  scheme @1 :FiniteDifferenceScheme = centralTwoPoint $mutability(mutableLive) $meta(dimension = dimensionless, unit = none, cliFlags = ["--fd-scheme"]);
  extrapolationOrder @2 :UInt8 = 0 $mutability(mutableLive);
}

struct FiniteDifferencePatch {
  stepAngstrom @0 :OptFloat64;
  scheme @1 :OptFiniteDifferenceScheme;
  extrapolationOrder @2 :OptUInt8;
}

struct CellConfig {
  boxMatrixAngstrom @0 :List(Float64) $mutability(betweenStepsWithHook) $meta(dimension = length, unit = angstrom);
  periodic @1 :List(Bool) $mutability(restartOnly);
  triclinic @2 :Bool = false $mutability(restartOnly);
}

struct CellPatch {
  boxMatrixAngstrom @0 :OptFloat64List;
  periodic @1 :OptInt32List;
  triclinic @2 :OptBool;
}

enum NeighborMethod {
  autoSelect @0;
  cellListCpu @1;
  bruteForceCpu @2;
  cellListGpu @3;
  vesinBackend @4;
}

struct OptNeighborMethod {
  union {
    unset @0 :Void;
    value @1 :NeighborMethod;
  }
}

struct NeighborListConfig {
  cutoffAngstrom @0 :Float64 = 5.0 $mutability(betweenStepsWithHook) $meta(dimension = length, unit = angstrom, bounds = (hasMinExclusive = true, minExclusive = 0.0));
  skinAngstrom @1 :Float64 = 1.0 $mutability(betweenStepsWithHook) $meta(dimension = length, unit = angstrom, bounds = (hasMinInclusive = true, minInclusive = 0.0));
  method @2 :NeighborMethod = autoSelect $mutability(betweenStepsWithHook);
  bothWays @3 :Bool = true $mutability(restartOnly);
  excludeSelf @4 :Bool = true $mutability(restartOnly);
  bruteForce @5 :Bool = false $mutability(betweenStepsWithHook);
  useCovalentRadii @6 :Bool = false $mutability(betweenStepsWithHook);
  covalentScale @7 :Float64 = 1.3 $mutability(betweenStepsWithHook) $meta(dimension = dimensionless, unit = none, bounds = (hasMinExclusive = true, minExclusive = 0.0));
  chunkSize @8 :UInt32 = 0 $mutability(mutableLive);
}

struct NeighborListPatch {
  cutoffAngstrom @0 :OptFloat64;
  skinAngstrom @1 :OptFloat64;
  method @2 :OptNeighborMethod;
  bothWays @3 :OptBool;
  excludeSelf @4 :OptBool;
  bruteForce @5 :OptBool;
  useCovalentRadii @6 :OptBool;
  covalentScale @7 :OptFloat64;
  chunkSize @8 :OptUInt32;
}
