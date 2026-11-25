Sync the vendored mopacc C ABI header in MOPACPot with upstream mopacc,
adding missing fields to MopacCParams and MopacCResult to prevent stack
corruption on by-value result returns. Verify struct sizes and field
offsets at compile and load time, and add an upstream-layout canary test.
