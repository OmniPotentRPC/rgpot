<!-- Copyright 2023--present rgpot developers -->

Periodic neighbour bins and the fast tile walk share one folded coordinate, so an atom outside the cell still meets its neighbours when a side has five or more bins. A pair vector is taken from the original positions, so wrapping a molecule into a cell no longer perturbs its energy.
