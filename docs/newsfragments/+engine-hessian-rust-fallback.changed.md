`EnginePotential::hessian` in the Rust loader picks its source as
`rgpot_engine_hessian` in the C table does: the dense hook when the engine
exports it and advertises `analytic_hessian`, otherwise one `rgpot_engine_hvp`
call per unit direction when it advertises `analytic_hvp`, otherwise
`EngineError::Unsupported` naming why each source is missing. An engine with
only an analytic Hessian-vector product now has a dense Hessian in Rust, and
`EnginePotential::into_potential` registers a dense-Hessian callback for it.
