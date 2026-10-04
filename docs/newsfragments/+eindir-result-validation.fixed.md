The eindir callbacks validate what a force callback returns with
`RGPOT_SUCCESS` before reading it: a null forces tensor or data pointer, a
wrong dtype, rank or element count, a non-finite force and a non-finite
energy are refused with `EINDIR_INVALID_PARAMETER` and a `rgpot_last_error()`
message, and the gradient output is left untouched. `n_atoms * 3` overflow is
refused at construction.
