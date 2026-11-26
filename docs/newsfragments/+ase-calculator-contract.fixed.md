The in-process ASE calculator follows the ASE calculator contract on five
points. calculate_many omits stress when the cell has fewer than three
nonzero lattice vectors, and raises PropertyNotImplementedError if stress
was requested. numerical_stress and stress_step are calculator parameters:
set updates them and todict round-trips a non-default value. calculate
with no properties list fills every implemented property. ase.db stores
the name rgpot. An empty system_changes list replaces results, so a key
this call did not produce does not remain.
