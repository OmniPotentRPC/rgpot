An eindir `value_and_gradient` whose energy fails now runs the potential callback once, not twice.
The fused cache keeps the failure, so the gradient asked for next at the same positions returns the same error without a second callback.
Before, a failed energy cleared the cache, and the gradient ran the engine again on a geometry already known to fail (for xtb, a second SCC that does not converge).
