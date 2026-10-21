The minimum-image fold for non-orthogonal cells runs in double precision.
Where long double is wider than double, cells that fail the double
condition test and displacements large enough to lose the image margin
use a double-double search. Where long double is double, the fold keeps
the historical arithmetic. The folded vector stays within 1e-9 of the
previous long-double result, and cells that were rejected as
ill-conditioned still throw.
