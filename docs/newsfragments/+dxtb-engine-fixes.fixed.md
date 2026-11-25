The dxtb engine now takes its Python executable from `BuiltinParams.library`
before `RGPOT_DXTB_PYTHON`, accepts the `gfn1-xtb` and `gfn2-xtb` method
spellings, and reports a charge or spin it cannot take as return code 3 without
changing the configuration it holds. Its worker socket is closed on exec, and
it starts correctly when descriptors 0 and 1 are free.
