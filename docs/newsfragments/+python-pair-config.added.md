`rgpot.LJPot` takes the `LJConfig` fields as keywords (`u0`, `cutoff`, `psi`,
`switch_width`), with the C++ defaults, so Python can turn on the C^2 cutoff
switch. An `ExprPot` child given as a dict reads the same keys for `"lj"` and
`De`, `a`, `re`, `cutoff` and `switch_width` for `"morse"`. Both pair kinds
were built from their defaults before.
