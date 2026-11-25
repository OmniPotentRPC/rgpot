`Neighbours` counts the cells of a grid in 64-bit arithmetic and bins a box
whose grid has more cells than an `int` can name as one cell.
A periodic box of 3e9 Å with a 1 Å search used to overflow the cell count and
throw `std::length_error` instead.
