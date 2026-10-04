`PotentialCache` counts computed and served evaluations per handle
(`counts()`, `reset_counts()`). The same split is available on eindir
potentials through `rgpot_potential_eval_counts` and
`rgpot_potential_reset_eval_counts`, and from Python as
`PotentialCache.counts()` (bound when built with the cache), so two runs under
different cache states report the same computed count.
