The C ABI minor of `rgpot-core` is now 11,
in `RGPOT_ABI_VERSION_MINOR`, for the additive
`rgpot_potential_set_hessian_byte_budget` export. A caller built
against minor 0 to 10 still matches; a caller built against minor 11
is refused by an older library.
