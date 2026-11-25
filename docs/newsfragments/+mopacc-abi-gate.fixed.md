MOPACPot now refuses a libmopacc whose C ABI or library ABI is not exactly
2, one that lacks mopacc_c_abi_version, mopacc_abi_version,
mopacc_result_size or mopacc_params_size, or one whose reported MopacCResult
or MopacCParams size does not match its own, before calling into the engine.
The load error names the expected and found values. A nonzero status from
mopacc_set_params also leaves the engine unloaded.
