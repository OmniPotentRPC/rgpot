# MOPACPot — consumer of the split mopacc engine

rgpot dlopens [`libmopacc.so`](https://github.com/OmniPotentRPC/mopacc).
This directory is the frontend only. OpenMOPAC lives in mopacc.

Params are packed `MopacCParams`. Default Hamiltonian is AM1; the
engine is every OpenMOPAC model plus SCF / relax / vibe.

```
app
  MOPACPot
    dlopen(MOPACC_LIBRARY / RGPOT_MOPACC_ENGINE / libmopacc.so)
      mopacc_set_params / mopacc_energy_gradient
        OpenMOPAC mopac_scf  model=4 AM1
```

| Variable | Role |
| --- | --- |
| `MOPACC_LIBRARY` | path to `libmopacc.so` |
| `RGPOT_MOPACC_ENGINE` | same |

ABI units: Hartree, Hartree/Bohr. Frontend: eV, eV/Å.

## Load check

mopacc C ABI 1 and 2 export the same compute symbols, but ABI 2 grew
`MopacCResult` (returned by value) and `MopacCParams`. Before any call
that returns a `MopacCResult`, the loader requires:

- `mopacc_c_abi_version()` and `mopacc_abi_version()` both exported and
  `>= 2`;
- `mopacc_result_size()` exported and equal to `sizeof(MopacCResult)`;
- `mopacc_params_size()` exported and equal to `sizeof(MopacCParams)`.

Otherwise the engine stays unloaded and `forceImpl` throws a load error
naming the expected and found values. A nonzero status from
`mopacc_set_params` also leaves the engine unloaded.
