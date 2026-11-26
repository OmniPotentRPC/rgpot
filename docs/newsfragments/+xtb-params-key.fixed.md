`XTBPot` now has a result-cache key that covers the method, charge, spin,
accuracy, electronic temperature, SCC iteration budget and SCC rescue
switch, read at each call. A result cached under one setting is no longer
served after a setter, or `rgpot_engine_set_scc_rescue` on the xtb engine,
changes it, so a rescued energy is not returned once the rescue is off.
