`rgpot::setThreadCount`, `rgpot::clearThreadCount`, `rgpot::ThreadCount` and
`PotentialBase::setThreadCount` set the thread count for a call. They are
opt-in: with none of them set, `threadsRequested` is still `OMP_NUM_THREADS`
(unset, empty or below 1 is one thread). A `ThreadCount` already on the stack
is not replaced by a potential's own setting, and a count below 1 throws
`std::invalid_argument`. Without OpenMP the request is reported by
`explicitThreadCount` and the call stays one thread. A potential binds its own
count while it evaluates, through `forceImpl` and `forcePerAtom` alike, and so
does the batch fan-out of a shared-instance potential, a `SumPot` for all of
its terms, and `HessianVectorCache` around the analytic and dual-number Hessian
and Hessian-vector hooks. A classical call still takes its slice count from
`teamSize`, so its results keep their bits on every count.
