The generic engine C ABI minor is now 5, in `RGPOT_ENGINE_ABI_MINOR` and
`ENGINE_ABI_MINOR`, for the additive `rgpot_engine_taylor_jet` entry point and
the trailing `RgpotEngineCaps.jet_order` field. The lowest engine minor a host
loads is a separate constant, `RGPOT_ENGINE_ABI_MINIMUM_MINOR` and
`ENGINE_ABI_MINIMUM_MINOR`, and stays 4, so an engine of minor 4 without the
jet still loads and reports `EngineError::Unsupported` only when the jet is
requested. `RgpotEngineAbiStamp::MINIMUM` is the stamp a load is checked
against.
