LDA TDA/RPA fxc forms `wv = w * rho * (v2rho2_0 + v2rho2_1)` and tiles
stage B at PySCF `BLKSIZE` (128). `--tda-rpa` pins host J from the
`get_j` call inside `gen_vind` / `gen_tdhf_operation`.
