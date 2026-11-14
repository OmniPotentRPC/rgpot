`-Dwith_eigen=true` falls back to the Eigen wrap when no system Eigen is found. Before, that fallback passed an unknown `subproject` keyword to `dependency()` and configure stopped with an error.
