`CBEPot` now evaluates the CH5 2008 surface through potlibc's `ch52008` instead
of the external cbe-pes library, and serves the engine gradient jet to order 3
through `potlibc_gradient_jet`, in one call for a whole batch, with the
forward-mode operation count and scale of the call, so the `cbe` backend of the
classical engine answers `rgpot_engine_taylor_jet` with `jet_order` 3. It also
evaluates energies and forces for a batch in one `potlibc_evaluate` call and
reports `batched`. Building with `with_cbe` needs a potlibc that provides
`potlibc_gradient_jet` and its `potlibc` pkg-config file. Bit goldens of the
energy, force, Hessian product and jet at the saddle and at a displaced
geometry pin the engine.
