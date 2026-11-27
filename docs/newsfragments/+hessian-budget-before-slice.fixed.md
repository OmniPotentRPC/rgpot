`rgpot_potential_hessian` now checks the overflow-safe byte
count of the matrix against the caller's dense-Hessian byte
budget before it forms a view of `hessian_out`, so a short buffer
declared through `rgpot_potential_set_hessian_byte_budget` returns
`RGPOT_INVALID_PARAMETER` without being touched.
