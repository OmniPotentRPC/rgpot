RgpotRpcCalculator now tears down pycapnp objects before leaving the KJ loop, preventing process aborts when close() runs after spawn().
