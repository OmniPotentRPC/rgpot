# asv suite

Timings come from asv.
Memory comes from memray through the `asv_bench_memray` plugin (`ray_*` methods).
Snakemake is the only driver.
Tool versions are pinned in `benchflow/requirements.txt` and repeated in `asv.conf.json`.

```sh
python -m pip install -r benchflow/requirements.txt
snakemake -c 1
python -m unittest benchflow.test_gate
```

`benchflow/config.yaml` sets `rounds` (10) and `runs` (10).
Those runs of `main` on one process measure the noise.
Each run uses `--interleave-rounds` and does not pass `--quick` or `--skip-existing`.
Two runs of the same commit give a calibration ratio: the median of the second run over the median of the first.
With MAD for the median absolute deviation of the calibration ratios, the relative threshold is

```text
max(0.05, 3 * MAD, the 99th percentile of |ratio - 1|, the maximum |ratio - 1|)
```

so the gate is at least as wide as every calibration ratio.
`factor` is `1 + relative_threshold`.
Each benchmark keeps its own factor.

Each run writes asv results under `benchflow/out/results/<run>`.
From those results, the exporter takes the one result for the run's commit, machine, and environment.
It fails when no result or more than one result matches.

Every run uses the machine name `bench-runner`.
Its `os`, `arch`, `cpu`, `num_cpu`, and `ram` fields are fixed generic values.
That record is in `benchflow/out/home/.asv-machine.json`, the asv machine file.
The machine check reads only the `bench-runner` record and ignores every other record in that file.

`snakemake -c 1 pr_gate` is the PR gate.
`base` and `head` must be distinct revisions.
The gate runs one `asv continuous --interleave-rounds` for those revisions, writes `benchflow/out/pr_compare.txt`, and exits with the per-benchmark result.
A row fails when its head/base median ratio is above its own factor.
Equal revisions do not run a comparison.

`warmup_time` drops calls before the first sample.
Those warm calls leave the timed geometry alone.
Each `evaluate_lj` call builds a new calculator and does not write the coordinate arrays.
`LJPot` is one object, and setup evaluates it once to fill the pair list.
The timed calls then repeat that same geometry, and the arrays from setup are not replaced.

The rgpot wheel is the Lennard-Jones binding.
`benchflow/build_wheel.py` passes meson `setup-args` after the pyproject list so the optional Fortran kernels, dispersion corrections, and metatomic engine stay off.
`patchelf` must be on `PATH`, because meson-python uses it to rewrite the run-time search path (RPATH).
`CC` and `CXX` are pinned to `gcc` and `g++`.
The build needs Python's development headers.
