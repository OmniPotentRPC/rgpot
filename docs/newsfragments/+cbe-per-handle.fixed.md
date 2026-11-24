`CBEPot` serializes potlibc calls on one handle. Distinct handles may run at the same time, and the classical engine reports per-instance reentrancy for CBE.
