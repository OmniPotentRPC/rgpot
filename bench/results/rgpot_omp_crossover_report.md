# rgpot OpenMP crossover sweep

## Measurement-only threshold-1 patch (not committed)

```diff
diff --git a/CppCore/rgpot/Slices.hpp b/CppCore/rgpot/Slices.hpp
index 7d27138d..77344d58 100644
--- a/CppCore/rgpot/Slices.hpp
+++ b/CppCore/rgpot/Slices.hpp
@@ -73,7 +73,7 @@ struct Range {
 /// costs more than such a call: on 8 threads a neighbour search of 64
 /// atoms, and a Sutton-Chen call on 55 to 80 atoms, took several times
 /// their one-thread time.
-inline constexpr std::size_t kTeamMinAtoms = 8192;
+inline constexpr std::size_t kTeamMinAtoms = 1;
 
 /// Slices for a call over `n` atoms: one below kTeamMinAtoms or when
 /// threadsRequested() is one, else threadsRequested(), at most `n`.
diff --git a/CppCore/rgpot/nlist/NeighbourList.cc b/CppCore/rgpot/nlist/NeighbourList.cc
index 1512efcc..083011ec 100644
--- a/CppCore/rgpot/nlist/NeighbourList.cc
+++ b/CppCore/rgpot/nlist/NeighbourList.cc
@@ -668,7 +668,7 @@ void build(Table &table, const double *positions, std::size_t n, const double *b
   // A handful of atoms is cheaper as a direct pair loop. The cell list
   // spends its time building bins, and a sheared cell of this size has
   // only a few bins, so the stencil visits every pair anyway.
-  if (!clustered(query) && n < 512) {
+  if (!clustered(query) && n < 1) {
     Columns cols;
     const bool full = full_shell(query);
     const int stencil =
```

## Per-potential crossover (threshold-1 build)

|Potential|Stage|Threads|Crossover atoms|Speedup at crossover|p-value|
|---|---|---:|---:|---:|---:|
|eam|build|2|256|1.96317|1.82923e-05|
|eam|build|8|256|4.30634|1.82923e-05|
|eam|force|2|256|1.08547|1.82923e-05|
|eam|force|8|256|2.56502|1.82923e-05|
|emt|build|2|256|1.98823|1.82923e-05|
|emt|build|8|256|4.70034|1.82923e-05|
|emt|force|2|256|1.04707|1.82923e-05|
|emt|force|8|256|1.06415|1.82923e-05|
|lj|build|2|256|1.98158|1.82923e-05|
|lj|build|8|256|6.6757|1.82923e-05|
|lj|force|2|256|1.09121|1.82923e-05|
|lj|force|8|256|4.07613|1.82923e-05|
|morse|build|2|256|1.48963|1.82923e-05|
|morse|build|8|256|4.73362|1.82923e-05|
|morse|force|2|256|1.43889|1.82923e-05|
|morse|force|8|256|3.69767|1.82923e-05|
|sutton-chen|build|2|256|1.82109|1.82923e-05|
|sutton-chen|build|8|256|4.65902|1.82923e-05|
|sutton-chen|force|2|none|n/a|n/a|
|sutton-chen|force|8|256|1.9224|1.82923e-05|
|sw|build|2|256|1.58415|1.82923e-05|
|sw|build|8|256|4.3727|1.82923e-05|
|sw|force|2|256|1.5412|1.82923e-05|
|sw|force|8|256|3.56951|1.82923e-05|
|tersoff|build|2|256|1.0676|0.000553089|
|tersoff|build|8|256|1.62942|1.82923e-05|
|tersoff|force|2|256|1.56748|1.82923e-05|
|tersoff|force|8|256|4.21351|1.82923e-05|
|zbl|build|2|256|1.02522|1.82923e-05|
|zbl|build|8|256|1.31508|1.82923e-05|
|zbl|force|2|4096|1.03509|1.82923e-05|
|zbl|force|8|256|1.69383|1.82923e-05|

## 1-thread throughput cost: threshold-1 vs stock

|Potential|Atoms|Force throughput ratio|Build throughput ratio|
|---|---:|---:|---:|
|eam|256|0.988833|0.905223|
|eam|512|1.00144|1.00315|
|eam|1024|1.00185|1.00083|
|eam|2048|0.999596|1.00389|
|eam|4096|1.00305|1.00233|
|eam|8192|0.999873|1.00035|
|eam|16384|1.00093|0.996179|
|emt|256|0.978418|0.673164|
|emt|512|1.00562|1.00676|
|emt|1024|1.00394|1.00274|
|emt|2048|1.00447|1.00557|
|emt|4096|1.00638|1.00367|
|emt|8192|0.998185|1.00177|
|emt|16384|0.997353|0.980387|
|lj|256|1.01576|0.811398|
|lj|512|0.991701|1.00629|
|lj|1024|1.01085|1.00482|
|lj|2048|1.01087|1.00435|
|lj|4096|1.01259|1.00291|
|lj|8192|0.992082|1.00239|
|lj|16384|0.995584|0.998481|
|morse|256|1.02928|8.13164|
|morse|512|0.975532|1.00581|
|morse|1024|0.983145|1.00546|
|morse|2048|0.982981|1.0018|
|morse|4096|0.989438|1.00597|
|morse|8192|0.982712|0.996152|
|morse|16384|0.987946|0.991789|
|sutton-chen|256|0.98015|22.918|
|sutton-chen|512|0.934797|1.00985|
|sutton-chen|1024|0.950722|1.01193|
|sutton-chen|2048|0.945865|0.966416|
|sutton-chen|4096|0.955812|0.964674|
|sutton-chen|8192|0.94189|0.965448|
|sutton-chen|16384|0.929683|0.943347|
|sw|256|0.953939|16.8605|
|sw|512|0.959907|1.00049|
|sw|1024|0.955847|1.01417|
|sw|2048|0.953871|1.01066|
|sw|4096|0.959273|0.98757|
|sw|8192|1.00943|0.976427|
|sw|16384|1.01167|0.970807|
|tersoff|256|0.962803|16.0795|
|tersoff|512|0.97054|1.00037|
|tersoff|1024|0.96568|0.999462|
|tersoff|2048|0.963278|0.998219|
|tersoff|4096|0.969427|1.00967|
|tersoff|8192|0.998287|1.00478|
|tersoff|16384|0.999732|0.99883|
|zbl|256|0.987335|14.1378|
|zbl|512|0.983435|1.00959|
|zbl|1024|0.986538|1.00898|
|zbl|2048|0.9764|1.00908|
|zbl|4096|0.957446|1.01173|
|zbl|8192|0.990628|1.00915|
|zbl|16384|0.995651|1.01327|

## Notes requested

- EMT force path does not call `forEachSlice`; it runs through `EMTPot::forceImpl` direct calls (`CppCore/rgpot/EMT/EMTPot.cc:39-78`).
- Tersoff and ZBL neighbour build rows are from the same internal builder (`rgpot::nlist::build`, `CppCore/rgpot/nlist/NeighbourList.cc:656-973`).
- Prior report text for Tersoff build scaling is corrected here by using measured build-only crossover and speedup tests.

## Requested follow-up findings

- EMT force stage remains effectively serial in this sweep: at 16384 atoms the stock force-only speedup is `1.00017x` (2 threads) and `1.00285x` (8 threads); threshold-1 build is `1.00202x` at 8 threads. This matches the code path above (no `forEachSlice` in `EMTPot::forceImpl`).
- Tersoff and ZBL neighbour-build scaling at 16384 atoms is weak in the stock build: Tersoff `1.1696x` and ZBL `1.1852x` at 8 threads (vs 1 thread). Both use the same internal builder; the measured cause is low per-thread useful build work at this size, with build still dominating total step time (`84.16%` for Tersoff and `93.97%` for ZBL at 8 threads).
- Sutton-Chen force-only slowdown at 2 threads is confirmed (`0.8068x` at 16384). Build time improves (`1.6094x`) while force time regresses, so the regression is in the force stage and not in neighbour building. `perf` counters were requested but unavailable on this host (`perf not installed`).
- LJ force-only at 2 threads is slightly faster (`1.0560x` at 16384); this is consistent with modest parallel gain in force while neighbour build still dominates wall time. `perf` counters were requested but unavailable on this host (`perf not installed`).

## Reproduction commands

- `neighbour_build_share`: `OMP_NUM_THREADS=8 OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES='{0},{1},{2},{3},{4},{5},{6},{7}' GOMP_CPU_AFFINITY='0,1,2,3,4,5,6,7' taskset -c 0-7 ./build-crossover-stock/bench/rgpot_omp_worker --potential lj --atoms 16384 --threads 8 --samples 12 --sample-seconds 0.02`
- `sutton_chen_two_thread_slowdown`: `OMP_NUM_THREADS=2 OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES='{0},{1}' GOMP_CPU_AFFINITY='0,1' taskset -c 0,1 ./build-crossover-th1/bench/rgpot_omp_worker --potential sutton-chen --atoms 16384 --threads 2 --samples 12 --sample-seconds 0.02`
- `emt_force_serial`: `OMP_NUM_THREADS=8 OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES='{0},{1},{2},{3},{4},{5},{6},{7}' GOMP_CPU_AFFINITY='0,1,2,3,4,5,6,7' taskset -c 0-7 ./build-crossover-th1/bench/rgpot_omp_worker --potential emt --atoms 16384 --threads 8 --samples 12 --sample-seconds 0.02`
- `tersoff_build_scaling`: `OMP_NUM_THREADS=8 OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES='{0},{1},{2},{3},{4},{5},{6},{7}' GOMP_CPU_AFFINITY='0,1,2,3,4,5,6,7' taskset -c 0-7 ./build-crossover-stock/bench/rgpot_omp_worker --potential tersoff --atoms 16384 --threads 8 --samples 12 --sample-seconds 0.02`

## Optional perf counters

- sutton-chen-force-2thread: `{"t1": {"available": false, "reason": "perf not installed"}, "t2": {"available": false, "reason": "perf not installed"}}`
- lj-force-2thread: `{"t1": {"available": false, "reason": "perf not installed"}, "t2": {"available": false, "reason": "perf not installed"}}`
