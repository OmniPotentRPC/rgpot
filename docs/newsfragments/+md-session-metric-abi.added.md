`rgpot_md_langevin_result` ends with `constraint_metric`, the plane metric of
every step, and `MdLangevinOutcome::constraint_metric` returns it. Callers and
engines built against the shorter struct keep working.
