`rgpot_core::eam::mishin_cu` evaluates the analytic EAM1 copper of Mishin et al., Phys. Rev. B 63, 224106 (2001).
It returns the energy, the forces and the Cauchy stress of a periodic cell with lattice vectors as rows.
`rgpot_core::eam::mishin_cu_potential` wraps it as a `PotentialImpl` callback.
`cargo bench -p rgpot-core --bench mishin_cu` times one serial force call on 31, 107 and 255 atoms.
