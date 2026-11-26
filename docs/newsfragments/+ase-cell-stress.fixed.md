RgpotCalculator raises PropertyNotImplementedError for stress and per-atom
stresses when the cell has fewer than three nonzero lattice vectors, matching
ASE's LennardJones and EMT calculators. Energy and forces are unchanged.
