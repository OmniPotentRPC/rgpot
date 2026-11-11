# UMA cutoff v1.8.6

Throw edges stay in the branch denominator.
The only excluded file is the vendored vesin translation unit.
UmaLinkedCell.hpp is 201 of 201 lines and 76 of 76 raw branches.
UmaEdges.cc is 457 of 457 lines and 239 of 242 raw branches.
That branch figure is 98.76 percent.
UmaGrow.hpp is 44 of 44 lines and 14 of 14 raw branches.
UmaPot.hpp is 4 of 4 lines.
UmaEdges.hpp and UmaConfig.hpp have no executable lines.
UmaPot.cc is 324 of 378 lines and 370 of 820 raw branches.
The CPU run leaves the non-CPU device copies untaken.
vesin-single-build.cpp with the file included is 618 of 1111 lines.
Its raw branches are 304 of 964.
The exclusion check rejects any other file.

A refused thread start joins the workers that already began.
The build then runs on one thread.
fit_recorded_edges keeps every edge when want is not positive.
An empty range does not start workers.

The panel is one process.
It times v1.8.3, v1.8.4, and v1.8.6 on the same calls.
Each cell has 24 calls of each version.
CPUs 0 through 7 are pinned.
Each row prints the host from gethostname, the CPU model, and the load.
The table is CppCore/tests/uma_cutoff_bench.txt.

At one thread the v1.8.4 over v1.8.3 weak ratio has median 1.037.
The interval is 1.024 to 1.045.
The gap grows with the thread count.
At 65536 atoms and 8 threads the weak ratio has median 1.963.
The interval is 1.854 to 2.041.
All 24 pairs are slower.
The strong cell at the same size has median 1.889.
The interval is 1.833 to 1.972.
All 24 pairs are slower.

The v1.8.6 over v1.8.3 intervals are below 1 or contain 1.
No cell has an interval above 1.
The 65536 by 8 weak interval is 0.919 to 0.968.
The strong interval at 8 threads is 0.925 to 1.066.
That strong interval contains 1.
