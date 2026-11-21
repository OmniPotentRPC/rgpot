<!-- Copyright 2023--present rgpot developers -->

Neighbour lists for triclinic and rotated periodic cells no longer omit pairs when cell binning is active. A sheared cell with both periodic and open axes now bins periodic directions by cell height and open directions by the atom span instead of using one all-pairs bin. A rotated or sheared cell whose diagonal entry on a periodic axis is zero or negative, such as a cube rotated by 90, 120 or 180 degrees, is binned and folded as a general cell and stays periodic along that axis.
