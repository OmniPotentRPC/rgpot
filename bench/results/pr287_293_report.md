# Pre-landing Performance Report: PR #287 and PR #293

## Environment and Protocol

- Base develop commit: `4aa93511d35c1cc75a93b0b372eb80c55f1365d8`
- Compiler: `g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`
- Build configuration: `release (meson -Dbuildtype=release -Dwith_eam_table=true -Dwith_fast_nlist=enabled)`
- CPU model: `Intel(R) Xeon(R) Processor` (`8` physical cores)
- Pinning (1 thread): `taskset` to 1 core, all thread env vars set to `1`, `OMP_PROC_BIND=close`, child affinity verified
- Pinning (multi-thread): `taskset` to exact core count (`2/4/8`), `OMP_NUM_THREADS=T`, `OMP_PROC_BIND=close`, explicit `OMP_PLACES`
- Sampling: pilot `14` rounds, adaptive `n` up to `48` targeting relative SE <= `3%` on `head/control` ratio; no `--quick`, no warm-up discards beyond bench's own, no rescue runs
- Decision rule: `regression` if 95% bootstrap CI of `head/control` lies strictly above the A/A band and exact Mann-Whitney `p < 0.01` after Bonferroni across PR cells; `win` by mirror rule; otherwise `neutral`

## Built Arms (Commit and Tree SHAs)

|PR|Arm|Commit SHA|Tree SHA|No `-march=native`|No `-ffast-math`|
|---|---|---|---|---|---|
|#287|control|`8b55e633109da481951bd0e4343918d37cee7fd0`|`f3b35c665dc206b668ee03e12c542653cbdc5260`|True|True|
|#287|control_aa|`8b55e633109da481951bd0e4343918d37cee7fd0`|`f3b35c665dc206b668ee03e12c542653cbdc5260`|True|True|
|#287|head|`1547b9091cbf42c1aafc52aa0a42c35be5629051`|`8c9af4f830f6c78be40cd7b6aa5a64bb092ff18b`|True|True|
|#293|control|`4aa93511d35c1cc75a93b0b372eb80c55f1365d8`|`65b870aedf6dcaf0ac8bfbacd70aa2728262d2bb`|True|True|
|#293|control_aa|`4aa93511d35c1cc75a93b0b372eb80c55f1365d8`|`65b870aedf6dcaf0ac8bfbacd70aa2728262d2bb`|True|True|
|#293|head|`a307ad422e460ce3956032d725917692f9b02af3`|`e89f138678d6cfe017af43449e4a11cb797aa080`|True|True|

## Per-PR Verdict Summary

|PR|Title|Cells|Wins|Regressions|Neutral|Verdict|
|---|---|---:|---:|---:|---:|---|
|#287|fix(nlist): multi-image neighbour lists for small cells (r5b)|20|1|0|19|**win**|
|#293|fix(sutton-chen): parallel forces under a finite term limit (r5)|16|3|0|13|**win**|

## PR #287: Large-cell Comparative Results (`head` vs `control` and `control_aa`)

|Potential|Atoms|Rounds|Control Med (ns)|Control IQR (ns)|A/A Med (ns)|Head Med (ns)|Head IQR (ns)|Head/Ctrl Ratio [95% CI]|A/A Band|Exact MW p|Bonf p|Verdict|
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
|nlist_build|32|14|69485|1717|69979|69950|1744|1.007 [0.9887, 1.026]|[0.9681, 1.033]|0.1499|1|neutral|
|nlist_build|256|14|4.0618e+06|1.271e+05|4.083e+06|4.0737e+06|5.902e+04|1.003 [0.9895, 1.024]|[0.9724, 1.028]|0.4824|1|neutral|
|nlist_build|1024|14|3.4338e+06|1.993e+05|3.5214e+06|3.5426e+06|1.975e+05|1.032 [0.9704, 1.061]|[0.9426, 1.061]|0.2456|1|neutral|
|nlist_build|4096|14|1.7785e+07|1.621e+05|1.7803e+07|1.7868e+07|7.044e+05|1.005 [0.9968, 1.035]|[0.9886, 1.012]|0.1371|1|neutral|
|eam|32|14|19951|133.1|19938|19915|157.6|0.9982 [0.9939, 1.005]|[0.9946, 1.005]|0.5714|1|neutral|
|eam|256|14|2.8942e+05|3386|2.8724e+05|2.9156e+05|7871|1.007 [0.9932, 1.023]|[0.9874, 1.013]|0.2649|1|neutral|
|eam|1024|14|1.0977e+06|1.974e+04|1.0998e+06|1.0962e+06|8684|0.9986 [0.985, 1.005]|[0.9879, 1.012]|0.5112|1|neutral|
|eam|4096|14|4.291e+06|5.307e+04|4.2753e+06|4.2827e+06|5.632e+04|0.9981 [0.9886, 1.007]|[0.983, 1.017]|0.6673|1|neutral|
|morse|32|48|6964.6|198.2|7004.4|6964.7|75.15|1 [0.9947, 1.006]|[0.987, 1.013]|0.8013|1|neutral|
|morse|256|14|1.2989e+05|2448|1.2959e+05|1.3031e+05|8401|1.003 [0.9895, 1.065]|[0.9501, 1.053]|0.5112|1|neutral|
|morse|1024|14|5.5886e+05|2.306e+04|5.5856e+05|5.6529e+05|2.575e+04|1.012 [0.9741, 1.048]|[0.9651, 1.036]|0.1936|1|neutral|
|morse|4096|14|2.1783e+06|1.272e+05|2.2066e+06|2.2525e+06|7.432e+04|1.034 [0.9903, 1.056]|[0.9606, 1.041]|0.1499|1|neutral|
|lj|32|14|7317.1|134.2|7322.7|7294.9|48.51|0.997 [0.9844, 1.005]|[0.9848, 1.015]|0.3287|1|neutral|
|lj|256|14|1.7274e+05|3637|1.7288e+05|1.7283e+05|2390|1 [0.9814, 1.009]|[0.9828, 1.018]|0.6027|1|neutral|
|lj|1024|14|7.0542e+05|2.111e+04|7.0687e+05|7.0465e+05|2.822e+04|0.9989 [0.9707, 1.03]|[0.9712, 1.03]|0.91|1|neutral|
|lj|4096|14|2.6987e+06|2.717e+04|2.7003e+06|2.7038e+06|2.554e+04|1.002 [0.9952, 1.01]|[0.9874, 1.013]|0.5714|1|neutral|
|eam_table|108|14|1.1098e+05|1766|1.1096e+05|1.1139e+05|3579|1.004 [0.9797, 1.017]|[0.9866, 1.014]|1|1|neutral|
|eam_table|256|14|3.2037e+05|1.49e+04|3.2167e+05|3.0426e+05|1.67e+04|0.9497 [0.9158, 0.9943]|[0.9583, 1.044]|0.004162|0.08324|neutral|
|eam_table|1024|14|1.1966e+06|1.853e+04|1.1932e+06|1.1664e+06|1.685e+04|0.9747 [0.9622, 0.9842]|[0.9856, 1.015]|0.0002128|0.004256|win|
|eam_table|4096|14|4.5885e+06|5.554e+04|4.5397e+06|4.4928e+06|3.41e+04|0.9791 [0.9702, 0.9853]|[0.9799, 1.02]|0.0002128|0.004256|neutral|

## PR #287: Small-cell Multi-image Cost (`EAMTablePot`, control rejects `edge <= 2*cutoff`)

|Cell|Atoms|Rounds|Head Med (ns)|Head IQR (ns)|Head ns/atom|Cost vs 108-atom ns/atom|
|---|---:|---:|---:|---:|---:|---:|
|eam_table|4|14|18360|280.3|4589.9|4.45x|
|eam_table|8|14|49038|464.6|6129.8|5.943x|
|eam_table|16|14|1.3645e+05|2150|8527.9|8.268x|
|eam_table|32|14|3.6967e+05|2564|11552|11.2x|
|eam_table_tri3|3|14|10373|241|3457.8|3.352x|

## PR #287: Small-cell Correctness Verification

|Potential|Atoms|Min Face / (2*Cut)|Control Status|Reference|Max |dE||Max |dF||Pairs (Got/Ref)|Notes|
|---|---:|---|---|---|---|---|---|---|
|eam_table|4|0.365|rejected: error: EAM table: cutoff exceeds half a face spacing|replicated_supercell_minimum_image|4.21e-13|1.855e-15|0/0|rep=3x3x3|
|eam_table|8|0.365|rejected: error: EAM table: cutoff exceeds half a face spacing|replicated_supercell_minimum_image|1.386e-12|2.772e-15|0/0|rep=3x3x2|
|eam_table|16|0.365|rejected: error: EAM table: cutoff exceeds half a face spacing|replicated_supercell_minimum_image|8.029e-13|3.061e-15|0/0|rep=3x2x2|
|eam_table|32|0.73|rejected: error: EAM table: cutoff exceeds half a face spacing|replicated_supercell_minimum_image|1.165e-11|3.01e-15|0/0|rep=2x2x2|
|eam_table_tri3|3|0.384|rejected: error: EAM table: cutoff exceeds half a face spacing|replicated_supercell_minimum_image|4.086e-14|1.421e-14|0/0|rep=3x3x3|
|nlist_build|32|0.723|ok|brute_force_multi_image_pairs|0|0|672/672|strict half-list pair count vs brute-force multi-image|
|lj|32|0.794|ok|brute_force_minimum_image|0|0|0/0|minimum-image pair potential|
|morse|32|0.675|ok|brute_force_minimum_image|0|6.661e-16|0/0|minimum-image pair potential|
|eam|32|0.57|ok|head_vs_control_minimage|0|0|0/0|bitwise_identical|

## PR #293: Sutton-Chen Comparative Results (`head` vs `control` and `control_aa`)

|Atoms|Threads|Rounds|Control Med (ns)|Control IQR (ns)|A/A Med (ns)|Head Med (ns)|Head IQR (ns)|Head/Ctrl Ratio [95% CI]|A/A Band|Exact MW p|Bonf p|Verdict|
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
|256|1|14|43531|1205|43519|43545|2510|1 [0.9757, 1.045]|[0.9781, 1.022]|0.8036|1|neutral|
|256|2|14|43039|375|43087|43036|543|0.9999 [0.9902, 1.006]|[0.99, 1.01]|0.5714|1|neutral|
|256|4|14|43113|597.5|43350|42973|595.9|0.9968 [0.985, 1.002]|[0.9872, 1.013]|0.09387|1|neutral|
|256|8|14|43029|249.9|42961|42672|272|0.9917 [0.9888, 0.9974]|[0.9923, 1.008]|0.002963|0.04741|neutral|
|1024|1|14|1.7877e+05|2532|1.7831e+05|1.8003e+05|4793|1.007 [0.9942, 1.03]|[0.9857, 1.014]|0.2649|1|neutral|
|1024|2|14|1.8051e+05|3987|1.8063e+05|1.7867e+05|1649|0.9898 [0.9755, 1.001]|[0.9847, 1.016]|0.07666|1|neutral|
|1024|4|14|1.7856e+05|1523|1.7826e+05|1.7873e+05|1729|1.001 [0.9913, 1.007]|[0.9914, 1.009]|0.8388|1|neutral|
|1024|8|14|1.7874e+05|1774|1.7952e+05|1.7962e+05|4392|1.005 [0.9935, 1.019]|[0.9762, 1.024]|0.8743|1|neutral|
|4096|1|14|7.0757e+05|1.751e+04|7.0112e+05|6.9829e+05|6186|0.9869 [0.9747, 1.003]|[0.9786, 1.022]|0.02119|0.339|neutral|
|4096|2|14|7.2438e+05|1.523e+04|7.1861e+05|7.2567e+05|1.601e+04|1.002 [0.9837, 1.018]|[0.9812, 1.019]|0.8036|1|neutral|
|4096|4|14|7.2626e+05|7293|7.2659e+05|7.1714e+05|9077|0.9874 [0.9827, 1.001]|[0.9703, 1.031]|0.2456|1|neutral|
|4096|8|14|7.2371e+05|6498|7.3187e+05|7.218e+05|1.695e+04|0.9974 [0.9865, 1.014]|[0.9718, 1.029]|0.7345|1|neutral|
|10976|1|14|1.9189e+06|6.638e+04|1.8935e+06|1.8977e+06|2.95e+04|0.989 [0.9651, 1.005]|[0.9645, 1.037]|0.1139|1|neutral|
|10976|2|14|1.8717e+06|8.537e+04|1.8723e+06|8.4395e+05|3.325e+04|0.4509 [0.4366, 0.4672]|[0.9657, 1.035]|4.985e-08|7.977e-07|win|
|10976|4|34|1.0056e+06|3.584e+04|1.0049e+06|4.6497e+05|1.601e+04|0.4624 [0.4504, 0.4672]|[0.9732, 1.028]|7.029e-20|1.125e-18|win|
|10976|8|14|5.833e+05|3.125e+04|5.6032e+05|2.8102e+05|9813|0.4818 [0.4689, 0.5015]|[0.9338, 1.071]|4.985e-08|7.977e-07|win|

## PR #293: Force Agreement vs Control and Determinism

|Atoms|Threads|Head vs Control Bitwise|Head vs Control Max |dF||Head vs T=1 Bitwise|Head vs T=1 Max |dF||Repeat-Call Bitwise|Repeat-Call Max |dF||
|---:|---:|---|---:|---|---:|---|---:|
|256|1|yes|0|yes|0|yes|0|
|256|2|yes|0|yes|0|yes|0|
|256|4|yes|0|yes|0|yes|0|
|256|8|yes|0|yes|0|yes|0|
|1024|1|yes|0|yes|0|yes|0|
|1024|2|yes|0|yes|0|yes|0|
|1024|4|yes|0|yes|0|yes|0|
|1024|8|yes|0|yes|0|yes|0|
|4096|1|yes|0|yes|0|yes|0|
|4096|2|yes|0|yes|0|yes|0|
|4096|4|yes|0|yes|0|yes|0|
|4096|8|yes|0|yes|0|yes|0|
|10976|1|yes|0|yes|0|yes|0|
|10976|2|yes|0|yes|0|yes|0|
|10976|4|yes|0|yes|0|yes|0|
|10976|8|yes|0|yes|0|yes|0|

## PR #293: Finite Term-Limit Fallback Check (`atoms=10976, term_limit=10000`)

|Threads|Head Med (ns)|Control Med (ns)|Head/Ctrl Ratio|Head vs Control Bitwise|Max |dF||
|---:|---:|---:|---:|---|---:|
|1|2.1166e+06|2.1105e+06|1.003|yes|0|
|2|1.9884e+06|2.0142e+06|0.9872|yes|0|
|4|1.07e+06|1.0514e+06|1.018|yes|0|
|8|6.2119e+05|5.9304e+05|1.047|yes|0|

## Slot Alternation Audit `(slot, arm)` Counts

|PR|Slot 0 (ctrl / aa / head)|Slot 1 (ctrl / aa / head)|Slot 2 (ctrl / aa / head)|
|---|---|---|---|
|#287|109 / 100 / 105|98 / 110 / 106|107 / 104 / 103|
|#293|85 / 78 / 81|78 / 80 / 86|81 / 86 / 77|

