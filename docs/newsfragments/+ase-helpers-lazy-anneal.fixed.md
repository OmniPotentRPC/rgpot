`ase_helpers` imports with only ASE and numpy again, so `atoms_from_con` and `atoms_to_con` work without the compiled `rgpot._core`.
`AnnealRunner` and `AnnealOptimizer` are loaded from `rgpot.ase.anneal` on first use.
