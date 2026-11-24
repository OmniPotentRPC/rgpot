Add the optional `rgpot_engine_taylor_jet` entry point, the
`RgpotEngineDerivativeReport` roundoff report and the trailing
`RgpotEngineCaps.jet_order` capability to the generic engine C ABI. One call
serves a uniform batch of systems and returns the exact first `jet_order`
directional derivatives of the gradient along a line for each, the first block
being the Hessian-vector product, with the summed operation count and largest
scale as a roundoff estimate. An engine without the jet reports
`RGPOT_ENGINE_ERR_UNSUPPORTED`, and the Rust loader maps that, a missing symbol
or a lower `jet_order` to `EngineError::Unsupported` through
`EngineSession::taylor_jet` and `EnginePotential::taylor_jet`. The loader asks
for the capabilities once per instance. A potential serves the jet by deriving
`rgpot::GradientJet`.
