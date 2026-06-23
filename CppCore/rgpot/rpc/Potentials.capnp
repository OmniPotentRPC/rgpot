# @brief RPC schema for distributed potential evaluations.
#
# This schema defines the binary communication contract between the light client
# and the RPC server components. The design derives from the C-style structures
# in the eOn [1] project (at the v4 writeup).
#
# # References
# [1] eOn Development Team. eOn Documentation. https://eondocs.org.

@0xbd1f89fa17369103;

# @struct ForceInput
# @brief Input configuration for a potential energy evaluation.
# @field lengthUnit Unit expression for positions/box (default "angstrom").
# @field energyUnit Unit expression for energy output (default "eV").
# Unit strings are parsed by rgpot::units::unit_conversion_factor().
# Examples: "angstrom", "bohr", "eV", "hartree", "kJ/mol", "kcal/mol".
struct ForceInput {
  pos        @0 :List(Float64); # @brief Flat array of atomic coordinates [natoms * 3].
  atmnrs     @1 :List(Int32);   # @brief Array of atomic numbers [natoms].
  box        @2 :List(Float64); # @brief Simulation cell vectors [9] (row-major 3x3).
  lengthUnit @3 :Text = "angstrom"; # @brief Unit for positions and box vectors.
  energyUnit @4 :Text = "eV";       # @brief Unit for energy and forces output.
}

# @struct PotentialResult
# @brief Results returned from a potential energy evaluation.
struct PotentialResult {
  energy @0 :Float64;       # @brief The calculated potential energy.
  forces @1 :List(Float64); # @brief Flat array of atomic forces [natoms * 3].
}

# @interface Potential
# @brief The RPC interface for remote calculations.
interface Potential {
  # @brief Executes the potential and force calculation.
  # @param fip The input atomic configuration.
  # @return The resulting energy and force vector.
  calculate @0 (fip :ForceInput) -> (result :PotentialResult);
}
