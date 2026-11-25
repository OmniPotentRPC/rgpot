`ZBLPot` now evaluates orthogonal, sheared and rotated periodic cells through
the shared neighbour layer, and `caps().periodic` reports `true`. Any non-zero
cell counts as periodic. A malformed non-zero cell throws
`std::invalid_argument`, while a null or all-zero cell stays open with free
boundaries. The `ZBLPot` kernel version moves from 1 to 2, so cached `ZBLPot`
results from earlier versions no longer match. The potential declares pairwise
`locality()` and provides `densityStage` and `forceStage` hooks, so `pair_style
rgpot` can split `ZBL` across ranks. It also registers `ZBL` in
`rgpot::spec::build`.
