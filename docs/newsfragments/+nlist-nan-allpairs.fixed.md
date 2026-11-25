In an orthorhombic or fully open skewed cell, `rgpot::nlist::Neighbours` now
falls back to the all-pairs walk when any atom coordinate is non-finite, instead
of binning `NaN` or `-inf` coordinates into cell zero and missing partners in
non-adjacent cells across `Auto`, `Stored`, and `Stream` modes. A skewed cell
with a periodic axis still throws `std::invalid_argument` for a non-finite
coordinate and keeps the state of its previous update.
