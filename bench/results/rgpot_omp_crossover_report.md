# rgpot OpenMP crossover sweep

## Measurement-only patches (not committed)

### Team-only patch

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
```

### Fallback-only patch

```diff
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

## Per-potential crossover (team-only build)

n per Mann-Whitney test arm: 12.

|Potential|Stage|Threads|Crossover atoms|Speedup at crossover|p-value|
|---|---|---:|---:|---:|---:|
|eam|build|2|8192|1.5926|3.6980115e-07|
|eam|build|8|8192|4.66545|3.6980115e-07|
|eam|force|2|8192|1.25113|3.6980115e-07|
|eam|force|8|8192|3.79119|3.6980115e-07|
|emt|build|2|8192|1.5948|3.6980115e-07|
|emt|build|8|8192|4.6142|3.6980115e-07|
|emt|force|2|none|n/a|n/a|
|emt|force|8|none|n/a|n/a|
|lj|build|2|8192|1.58225|3.6980115e-07|
|lj|build|8|8192|4.59421|3.6980115e-07|
|lj|force|2|none|n/a|n/a|
|lj|force|8|8192|2.52631|3.6980115e-07|
|morse|build|2|8192|1.72846|3.6980115e-07|
|morse|build|8|8192|4.33747|3.6980115e-07|
|morse|force|2|8192|1.50481|3.6980115e-07|
|morse|force|8|8192|3.36017|3.6980115e-07|
|sutton-chen|build|2|8192|1.77783|3.6980115e-07|
|sutton-chen|build|8|8192|4.42715|3.6980115e-07|
|sutton-chen|force|2|none|n/a|n/a|
|sutton-chen|force|8|8192|3.14504|3.6980115e-07|
|sw|build|2|8192|1.79214|3.6980115e-07|
|sw|build|8|8192|3.38869|3.6980115e-07|
|sw|force|2|8192|1.37426|3.6980115e-07|
|sw|force|8|8192|5.36948|3.6980115e-07|
|tersoff|build|2|none|n/a|n/a|
|tersoff|build|8|none|n/a|n/a|
|tersoff|force|2|8192|1.52624|3.6980115e-07|
|tersoff|force|8|8192|5.66835|3.6980115e-07|
|zbl|build|2|none|n/a|n/a|
|zbl|build|8|none|n/a|n/a|
|zbl|force|2|none|n/a|n/a|
|zbl|force|8|8192|2.25164|3.6980115e-07|

## Fallback-only vs stock A/B at 1 thread (build time)

|Potential|Atoms|Cutoff|Shell|Clustered|Neighbors/atom (stock median)|Build-time ratio fallback/stock|Exact one-sided MW p (fallback faster)|n|
|---|---:|---:|---|---|---:|---:|---:|---:|
|eam|32|7.1|half|false|70|0.997518|0.14567133|12|
|eam|64|7.1|half|false|70|0.99937|0.421418|12|
|eam|128|7.1|half|false|70|0.998805|0.37764241|12|
|eam|256|7.1|half|false|70|0.999945|0.4887011|12|
|eam|384|7.1|half|false|70|0.997849|0.089091014|12|
|eam|512|7.1|half|false|70|1.00189|0.92001349|12|
|emt|256|7.1|half|false|112|1.00125|0.93615605|12|
|emt|384|7.1|half|false|112|0.997205|0.029828531|12|
|emt|512|7.1|half|false|112|0.998057|0.33532977|12|
|lj|32|2.5|half|false|88|1.00143|0.4887011|12|
|lj|64|2.5|half|false|88|1.0047|0.977549|12|
|lj|128|2.5|half|false|88|1.00345|0.86716521|12|
|lj|256|2.5|half|false|88|1.0001|0.4887011|12|
|lj|384|2.5|half|false|88|1.00037|0.44369297|12|
|lj|512|2.5|half|false|88|0.996266|0.06384395|12|
|morse|32|6|half|false|43|0.994744|0.03418072|12|
|morse|64|6|half|false|43|1.00254|0.7952448|12|
|morse|128|6|half|false|43|1.00181|0.92001349|12|
|morse|256|6|half|false|43|0.992383|0.44369297|12|
|morse|384|6|half|false|43|1.01617|0.55630703|12|
|morse|512|6|half|false|43|0.997962|0.071583888|12|
|sutton-chen|32|3|half|false|6|0.996131|0.27558358|12|
|sutton-chen|64|3|half|false|6|1.00262|0.76121237|12|
|sutton-chen|128|3|half|false|6|0.999379|0.60062733|12|
|sutton-chen|256|3|half|false|6|0.997555|0.2047552|12|
|sutton-chen|384|3|half|false|6|1.00127|0.35626754|12|
|sutton-chen|512|3|half|false|6|0.999926|0.23878763|12|
|sw|32|3.77118|half|false|14|0.999018|0.578582|12|
|sw|64|3.77118|half|false|14|0.999743|0.55630703|12|
|sw|128|3.77118|half|false|14|1.00368|0.77858341|12|
|sw|256|3.77118|half|false|14|1.00007|0.55630703|12|
|sw|384|3.77118|half|false|14|0.997112|0.22141659|12|
|sw|512|3.77118|half|false|14|1.00061|0.92001349|12|
|tersoff|32|3|half|false|2|0.997482|0.016620713|12|
|tersoff|64|3|half|false|2|0.99901|0.578582|12|
|tersoff|128|3|half|false|2|0.998881|0.22141659|12|
|tersoff|256|3|half|false|2|0.997029|0.5112989|12|
|tersoff|384|3|half|false|2|0.995764|0.25686129|12|
|tersoff|512|3|half|false|2|1.00161|0.77858341|12|
|zbl|32|5|half|false|2|0.997789|0.5112989|12|
|zbl|64|5|half|false|2|0.997815|0.23878763|12|
|zbl|128|5|half|false|2|1.00071|0.66467023|12|
|zbl|256|5|half|false|2|1.00638|0.98790787|12|
|zbl|384|5|half|false|2|0.999784|0.44369297|12|
|zbl|512|5|half|false|2|0.9997|0.46615025|12|

## 1-thread throughput cost: team-only vs stock

|Potential|Atoms|Force throughput ratio|Build throughput ratio|
|---|---:|---:|---:|
|eam|64|1.00141|0.995852|
|eam|128|0.999676|1.0004|
|eam|256|1.00073|0.999706|
|eam|512|0.998042|1.00162|
|eam|1024|1.00374|1.00081|
|eam|2048|1.0028|1.00074|
|eam|4096|1.00008|0.999469|
|eam|8192|1.00209|1.00096|
|eam|16384|0.997581|0.995198|
|emt|256|0.995293|1.00061|
|emt|512|0.997934|1.00087|
|emt|1024|0.993691|1.0032|
|emt|2048|0.998791|1.00172|
|emt|4096|1.0003|1.00099|
|emt|8192|1.0031|1.00045|
|emt|16384|0.998225|1.00146|
|lj|64|1.00429|1.00264|
|lj|128|1.00172|0.997724|
|lj|256|1.00296|0.999505|
|lj|512|0.996994|0.996824|
|lj|1024|1.00768|1.00001|
|lj|2048|0.999584|1.00091|
|lj|4096|1.01083|1.0005|
|lj|8192|0.989454|0.999353|
|lj|16384|1.00768|1.00821|
|morse|64|1.0095|0.999683|
|morse|128|0.996373|0.995527|
|morse|256|0.99828|1.00553|
|morse|512|0.997049|1.00404|
|morse|1024|1.01036|0.998223|
|morse|2048|0.994523|1.00126|
|morse|4096|0.994122|0.999291|
|morse|8192|1.0007|0.999924|
|morse|16384|1.0043|1.00924|
|sutton-chen|64|1.00638|1.00207|
|sutton-chen|128|1.00661|1.00173|
|sutton-chen|256|1.0047|1.00307|
|sutton-chen|512|1.00272|1.00202|
|sutton-chen|1024|0.99745|1.00116|
|sutton-chen|2048|0.998377|0.999738|
|sutton-chen|4096|1.00301|1.00006|
|sutton-chen|8192|1.00127|0.997015|
|sutton-chen|16384|1.01858|1.00227|
|sw|64|1.00297|1.00501|
|sw|128|0.998409|0.996159|
|sw|256|0.999967|1.00097|
|sw|512|0.999433|1.00038|
|sw|1024|0.999028|1.00103|
|sw|2048|1.0001|0.997644|
|sw|4096|0.997445|1.00222|
|sw|8192|1.00191|0.998866|
|sw|16384|0.99857|0.997197|
|tersoff|64|1.00075|1.00006|
|tersoff|128|1.00616|1.00092|
|tersoff|256|1.00276|1.00229|
|tersoff|512|0.998175|1.00005|
|tersoff|1024|1.00285|1.00141|
|tersoff|2048|1.00059|1.00097|
|tersoff|4096|1.00069|0.998642|
|tersoff|8192|1.00146|1.00121|
|tersoff|16384|1.00074|1.00064|
|zbl|64|0.995664|0.994227|
|zbl|128|0.998998|0.994101|
|zbl|256|1.00231|0.998216|
|zbl|512|1.00144|1.00126|
|zbl|1024|1.00499|1.00014|
|zbl|2048|1.00426|1.00124|
|zbl|4096|0.997621|1.0004|
|zbl|8192|0.996185|0.999431|
|zbl|16384|1.00224|1.00046|

## Notes requested

- EMT force path does not call `forEachSlice`; it runs through `EMTPot::forceImpl` direct calls (`CppCore/rgpot/EMT/EMTPot.cc:39-78`).
- Tersoff and ZBL neighbour build rows are from the same internal builder (`rgpot::nlist::build`, `CppCore/rgpot/nlist/NeighbourList.cc:656-973`) and the same query shape (`half`, non-clustered); the main measured difference across potentials is per-atom neighbour count from cutoff and lattice.
- One-sided MW p-values are exact under the null for n=12 vs n=12.

## Additional tightening notes

- EMT in fallback A/B starts at 256 atoms (smallest valid cell in this harness); 32/64/128 fail EMT geometry.
- Team-only 1-thread slice-path cost source (file:line): `SuttonChenPot.cc` switches to row/slice bookkeeping when `rows = n_size >= kTeamMinAtoms` and `by_row` (`CppCore/rgpot/SuttonChen/SuttonChenPot.cc:177-190`), while SW and Tersoff switch from direct-centre loops to `sumCentres` when `n_atoms >= kTeamMinAtoms` (`CppCore/rgpot/StillingerWeber/SWSi.cc:267-282`, `CppCore/rgpot/Tersoff/TersoffSiC.cc:381-396`).
- At 16384 atoms, Tersoff and ZBL have the same query shape but much lower neighbour density (`2` neighbours/atom median) than EAM/LJ/EMT (`70-112`), which explains why their neighbour-build speedups at 8 threads stay near `1.06x/1.16x` while dense lists scale much higher.

## Reproduction commands

- `neighbour_build_share`: `OMP_NUM_THREADS=8 OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES='{0},{1},{2},{3},{4},{5},{6},{7}' GOMP_CPU_AFFINITY='0,1,2,3,4,5,6,7' taskset -c 0-7 ./build-crossover-stock/bench/rgpot_omp_worker --potential lj --atoms 16384 --threads 8 --samples 12 --sample-seconds 0.02`
- `sutton_chen_two_thread_slowdown`: `OMP_NUM_THREADS=2 OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES='{0},{1}' GOMP_CPU_AFFINITY='0,1' taskset -c 0,1 ./build-crossover-team/bench/rgpot_omp_worker --potential sutton-chen --atoms 16384 --threads 2 --samples 12 --sample-seconds 0.02`
- `emt_force_serial`: `OMP_NUM_THREADS=8 OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES='{0},{1},{2},{3},{4},{5},{6},{7}' GOMP_CPU_AFFINITY='0,1,2,3,4,5,6,7' taskset -c 0-7 ./build-crossover-team/bench/rgpot_omp_worker --potential emt --atoms 16384 --threads 8 --samples 12 --sample-seconds 0.02`
- `tersoff_build_scaling`: `OMP_NUM_THREADS=8 OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES='{0},{1},{2},{3},{4},{5},{6},{7}' GOMP_CPU_AFFINITY='0,1,2,3,4,5,6,7' taskset -c 0-7 ./build-crossover-stock/bench/rgpot_omp_worker --potential tersoff --atoms 16384 --threads 8 --samples 12 --sample-seconds 0.02`

## Optional perf counters

- sutton-chen-force-2thread: `{"t1": {"available": false, "errno_hint": "ENOTSUP", "reason": "perf events not supported in this VM"}, "t2": {"available": false, "errno_hint": "ENOTSUP", "reason": "perf events not supported in this VM"}}`
- lj-force-2thread: `{"t1": {"available": false, "errno_hint": "ENOTSUP", "reason": "perf events not supported in this VM"}, "t2": {"available": false, "errno_hint": "ENOTSUP", "reason": "perf events not supported in this VM"}}`
- tersoff-zbl-build-8thread: `{"tersoff_t1": {"available": false, "errno_hint": "ENOTSUP", "reason": "perf events not supported in this VM"}, "tersoff_t8": {"available": false, "errno_hint": "ENOTSUP", "reason": "perf events not supported in this VM"}, "zbl_t1": {"available": false, "errno_hint": "ENOTSUP", "reason": "perf events not supported in this VM"}, "zbl_t8": {"available": false, "errno_hint": "ENOTSUP", "reason": "perf events not supported in this VM"}}`
