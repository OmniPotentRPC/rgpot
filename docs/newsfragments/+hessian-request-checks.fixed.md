
`rgpot_potential_hessian` refuses a dense matrix whose byte count does not fit in
`size_t`, or that exceeds a caller byte budget installed with
`rgpot_potential_set_hessian_byte_budget`. There is no built-in dimension or MiB
limit; a NULL budget pointer clears the limit. Replacing or clearing a
dense-Hessian callback waits until every in-flight call of the previous callback
finishes, so callers may release callback state as soon as the setter returns.
