`XTBDlopen` and `MetatomicDlopen` now have `setThreadCount`, `clearThreadCount`
and `threadCount`, so a host can treat them like the other potentials. Each
force call forwards the resolved count, including a surrounding `ThreadCount`
scope, to the engine through the new `rgpot_xtb_set_thread_count` and
`rgpot_mta_set_thread_count` calls. The xtb and metatomic engine C ABIs are now
version 2, and a frontend refuses an engine of version 1. `setThreadCount`
throws `std::invalid_argument` for a count below 1.
