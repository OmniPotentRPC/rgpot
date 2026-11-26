GFN-xTB evaluations say when the SCC needed the smeared-occupation rescue,
and the rescue can be switched off. `XTBPot` and `XTBDlopen` gain
`lastEvalFlags()`, `setSccRescue()` and `XTBConfig::scc_rescue`; with the
rescue off an SCC that does not converge throws `SccNotConverged` instead of
returning a rescued energy. The engine C ABI (minor 12) adds
`rgpot_engine_last_eval_flags`, `rgpot_engine_set_scc_rescue` and
`rgpot_engine_source_revision`, the XTB engine ABI adds the matching optional
`rgpot_xtb_*` symbols, the Rust loader adds `EvalFlags`,
`EnginePotential::force_flagged`, `last_eval_flags`, `set_scc_rescue` and
`source_revision`, and Python adds `rgpot.XTBDlopen` and
`rgpot.SccNotConverged`. The rescue stays on by default. Hosts built against
engine ABI minor 11 or older refuse engines built from this release.
