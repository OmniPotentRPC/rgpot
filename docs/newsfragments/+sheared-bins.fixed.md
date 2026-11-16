`nlist::Neighbours` bins a sheared periodic cell along its cell vectors, with bins no narrower than the cutoff plus the skin between opposite faces, so the 27 neighbouring bins hold every pair within that distance.
It used to bin along x, y and z, which loses pairs once the tilts grow: with tilts of 0.2 to 0.3 of the side, inside what LAMMPS allows, it found 83468 of the 93483 pairs within 5 Angstrom in a 4000-atom fcc cell.
The potentials on that list (LJ, LJCluster, Morse, Sutton-Chen, ZBL, EAM, EAMTable, Stillinger-Weber and Tersoff) now see every pair in such cells.
Orthogonal cells bin as before, and where the old bins found every pair a stored list on one thread walks the same pairs in index order and keeps its bits; a streaming walk follows the bins and agrees to rounding.
