The aarch64 Linux wheels build again.
The minimum-image fold takes `target("fma")` on x86-64 only.
That is the one target where `fma_hw_active()` can select the hardware body.
