MOPACPot now refuses a libmopacc older than C ABI 2, or one whose
mopacc_result_size does not match its own MopacCResult, before calling into
the engine. The load error names the expected and found values.
