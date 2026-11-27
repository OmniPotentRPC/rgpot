The generic engine C ABI minor is now 12, in `RGPOT_ENGINE_ABI_MINOR`
and `ENGINE_ABI_MINOR`, for the optional `rgpot_engine_hessian`
entry point and the trailing `RgpotEngineCaps.analytic_hessian`
capability flag. A caller built against an earlier minor still matches.
`rgpot_engine_hessian` writes the backend's dense Hessian when the backend
advertises `analytic_hessian`, assembles the matrix from analytic
Hessian-vector products when it advertises only `analytic_hvp`, and returns
`RGPOT_ENGINE_ERR_UNSUPPORTED` when it advertises neither.
