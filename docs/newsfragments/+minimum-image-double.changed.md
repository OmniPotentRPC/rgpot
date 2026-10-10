The minimum-image fold for non-orthogonal cells runs in double precision.
Cells that fail the double condition test, and displacements large enough
to lose the image margin, use a double-double search. The folded vector
stays within 1e-9 of the previous long-double result, and cells that were
rejected as ill-conditioned still throw.
