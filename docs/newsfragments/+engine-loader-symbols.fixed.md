`EnginePotential::load` now calls `rgpot_engine_available` before
`rgpot_engine_create` and refuses an engine that reports itself
unavailable with `EngineError::Unavailable`. Every resolved symbol is
checked for a null address, so a library that exports a null entry is
reported as a missing symbol instead of being called.
