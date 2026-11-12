# asv suite

Timings come from asv.
Memory comes from memray through the `asv_bench_memray` plugin (`ray_*` methods).
Snakemake is the only driver.
Tool versions are pinned in `benchflow/requirements.txt` and repeated in `asv.conf.json`.

```sh
python -m pip install -r benchflow/requirements.txt
snakemake -c 1
python -m unittest discover -s benchflow
```

`benchflow/config.yaml` sets `rounds` (10) and `runs` (10).
Those runs of `develop` on one process measure the noise.
Each run uses `--interleave-rounds` and does not pass `--quick` or `--skip-existing`.
Two runs of the same commit give a calibration ratio: the median of the second run over the median of the first.
With MAD for the median absolute deviation of the calibration ratios, the relative threshold is

```text
max(0.05, 3 * MAD, the 99th percentile of |ratio - 1|, the maximum |ratio - 1|)
```

so the upper bound is at least the widest calibration ratio.
It is also at least five percent above one.
Each benchmark keeps its own upper bound.

Each run writes asv results under `benchflow/out/results/<run>`.
From those results, the exporter takes the one result for the run's commit, machine, and environment.
It fails when no result or more than one result matches.

Every run uses the machine name `bench-runner`.
Its `os`, `arch`, `cpu`, `num_cpu`, and `ram` fields are fixed generic values.
That record is in `benchflow/out/home/.asv-machine.json`, the asv machine file.
The machine check reads only the `bench-runner` record and ignores every other record in that file.

`snakemake -c 1 pr_compare` compares two distinct revisions, `base` and `head`.
It runs one `asv continuous --interleave-rounds` for them and writes the table to `benchflow/out/pr_compare.txt`.
A row in that table fails when its head/base median ratio is above its own upper bound.
The result goes to `benchflow/out/pr_compare.status`, and `pr_compare` itself exits zero.
Equal revisions do not run a comparison, and their status is non-zero.
`pr_gate` exits with that status, so it fails on a slowdown.
Continuous integration should run `pr_gate`.

`warmup_time` drops calls before the first sample.
Those warm calls leave the timed geometry alone.
Each `evaluate_lj` call builds a new calculator and does not write the coordinate arrays.
`LJPot` is one object, and setup evaluates it once to fill the pair list.
The timed calls then repeat that same geometry, and the arrays from setup are not replaced.

The wheel is the installed package the timings call.
`LJClusterPot`, `MorsePot`, `WaterHPot`, and `LJPot` run the force implementations in that wheel's `librgpot`.
`benchflow/build_wheel.py` passes `-Dbuildtype=release` and `-Dwith_fortran_pots=enabled`.
Dispersion corrections and the metatomic engine stay off.
Meson's default buildtype is debug, so a configure that omits the flag is not this result.
The build keeps its meson tree and writes `meson introspect --buildoptions` beside the wheel.
`patchelf` must be on `PATH`, because meson-python uses it to rewrite the run-time search path (RPATH).
`CC` and `CXX` are pinned to `gcc` and `g++`.
The build needs Python's development headers and a Fortran compiler.
