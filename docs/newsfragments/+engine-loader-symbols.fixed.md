`EnginePotential::load` now calls `rgpot_engine_available` before
`rgpot_engine_create` when the plugin exports it, and refuses an engine that
reports itself unavailable with `EngineError::Unavailable`. A plugin that does
not export the symbol, or exports a null entry for it, is treated as available,
so such a plugin keeps loading. The symbol is the optional
`EngineSymbols::available` field, set with `EngineSymbols::with_available`, and
`EngineSymbols::new` keeps taking the four required symbols. Every other
resolved symbol is checked for a null address, so a library that exports a null
required entry is reported as a missing symbol instead of being called.
