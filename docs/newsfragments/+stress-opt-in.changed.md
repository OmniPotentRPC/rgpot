<!-- Copyright 2023--present rgpot developers -->

`ForceOut::stress` and `has_stress` are now written only when `ForceInput::compute_stress` asks for them. `LJPot`, `MorsePot`, `ZBLPot`, `StillingerWeberSi`, `TersoffSiC` and the Fortran `SWPot` used to fill the stress on every call; a force-only call now sets `has_stress` to 0 and the tensor to zero, with the same energy and forces. The `stress` fields of `SuttonChenConfig`, `EAMTableConfig` and `MEAMConfig` still request it on every call. In Python, `evaluate` returns a stress only with `compute_stress=True`.
