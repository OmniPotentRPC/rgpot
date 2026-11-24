The generic engine C ABI minor is now 11, in `RGPOT_ENGINE_ABI_MINOR` and
`ENGINE_ABI_MINOR`, for the additive `rgpot_engine_taylor_jet` entry point and
the trailing `RgpotEngineCaps.jet_order` field. A host serves any engine of the
same major and layout whose minor is at most its own, so an engine of minor 10
or earlier without the jet still loads and reports `EngineError::Unsupported`
only when the jet is requested.
