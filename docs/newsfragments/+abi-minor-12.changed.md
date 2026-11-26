The C ABI minor of `rgpot-core` is now 12, in `RGPOT_ABI_VERSION_MINOR`, for the
opt-in stress entry points (`rgpot_potential_calculate_stress`,
`rgpot_potential_set_stress_callback`, `rgpot_potential_supports_stress`, and
`rgpot_stress_out_create`). A caller built against minor 0 to 11
still matches; a caller built against minor 12 is refused by an older library.
