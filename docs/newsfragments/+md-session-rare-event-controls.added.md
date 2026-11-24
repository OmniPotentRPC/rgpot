Added the controls a rare-event sampler needs to the generic MD session
ABI: `rgpot_md_session_run_langevin` returns the Lagrange multiplier of a
hyperplane per step (the `dF/dc` integrand of PAFI and the force of
Bennett-Chandler), accepts reflecting or rejecting walls, and keeps its
seeded random stream across calls; `rgpot_md_session_reseed_velocities` draws
Maxwell-Boltzmann velocities, with `n . v = 0` on a plane, identically for
every engine; and the periodic box and the first force are set and read through
`rgpot_md_session_set_periodic_box`, `rgpot_md_session_get_periodic_box`,
`rgpot_md_session_initialize`, and the getters for positions, velocities,
forces and energies. The new callbacks follow `destroy` in the vtable, so an
engine built against the previous header still loads. `rgpot_core::md_session`
mirrors them with `MdSession::run_langevin` and the matching
Maxwell-Boltzmann draw.
