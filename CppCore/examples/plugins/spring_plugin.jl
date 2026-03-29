# rgpot plugin: harmonic spring potential in Julia.
#
# Demonstrates implementing an rgpot plugin in Julia using ccall-compatible
# functions. Julia can produce shared libraries via PackageCompiler.jl:
#
#   using PackageCompiler
#   create_sysimage(["SpringPlugin"]; sysimage_path="rgpot_spring.so",
#                    precompile_execution_file="precompile.jl")
#
# Alternatively, use Julia's C-callable function pointers with @cfunction
# and a thin C shim that dlopen's Julia and calls jl_eval_string().
#
# This file shows the Julia implementation. The actual .so wrapper would
# need PackageCompiler or a C shim similar to the Python bridge.
#
# E = 0.5 * k * sum(|r_i - r_i0|^2)
# F_i = -k * (r_i - r_i0)

module SpringPlugin

using LinearAlgebra

mutable struct SpringPotential
    k::Float64
    r0::Union{Nothing, Matrix{Float64}}
end

SpringPotential(; k::Float64=1.0) = SpringPotential(k, nothing)

function calculate!(
    pot::SpringPotential,
    positions::Matrix{Float64},     # 3 x n_atoms (column-major)
    atomic_numbers::Vector{Int32},
    cell::Matrix{Float64},          # 3 x 3
    forces::Matrix{Float64},        # 3 x n_atoms (output)
)::Float64
    if pot.r0 === nothing
        pot.r0 = copy(positions)
    end

    disp = positions .- pot.r0
    energy = 0.5 * pot.k * sum(disp .^ 2)
    forces .= -pot.k .* disp

    return energy
end

# -- Plugin instances --
const _instances = Dict{Int, SpringPotential}()
_next_id = Ref(1)

function plugin_create(config_json::String)::Int
    # Minimal JSON parsing (production code would use JSON3.jl)
    k = 1.0
    m = match(r"\"k\"\s*:\s*([0-9.eE+-]+)", config_json)
    if m !== nothing
        k = parse(Float64, m.captures[1])
    end
    pot = SpringPotential(k=k)
    id = _next_id[]
    _next_id[] += 1
    _instances[id] = pot
    return id
end

function plugin_destroy(handle::Int)
    delete!(_instances, handle)
end

function plugin_calculate(
    handle::Int,
    n_atoms::Int,
    positions_flat::Vector{Float64},    # length 3*n_atoms, row-major from C
    atomic_numbers::Vector{Int32},
    cell_flat::Vector{Float64},         # length 9, row-major from C
)::Tuple{Float64, Vector{Float64}}
    pot = _instances[handle]

    # Convert row-major C layout to column-major Julia
    pos = reshape(positions_flat, 3, n_atoms)
    cell = reshape(cell_flat, 3, 3)
    forces = zeros(Float64, 3, n_atoms)

    energy = calculate!(pot, pos, atomic_numbers, cell, forces)

    # Flatten forces back to row-major for C
    return (energy, vec(forces))
end

# Self-test
function main()
    pos = Float64[0 1 0; 0 0 1; 0 0 0]  # 3 x 3 (3 atoms)
    atm = Int32[1, 1, 1]
    cell = Float64[10 0 0; 0 10 0; 0 0 10]
    forces = zeros(Float64, 3, 3)

    pot = SpringPotential(k=2.0)
    e = calculate!(pot, pos, atm, cell, forces)
    println("Initial energy: $e (expect 0.0)")
    @assert abs(e) < 1e-12

    pos2 = pos .+ 0.1
    e2 = calculate!(pot, pos2, atm, cell, forces)
    println("Displaced energy: $e2 (expect 0.09)")
    println("Forces: $forces")
    @assert abs(e2 - 0.09) < 1e-12
    println("All checks passed.")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    SpringPlugin.main()
end
