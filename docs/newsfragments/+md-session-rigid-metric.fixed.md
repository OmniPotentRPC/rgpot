The plane multiplier of `ProjectedLangevin` includes the centripetal force that
holds rotating distance constraints, and the run records the metric
`Z = 1 / [(J M^-1 J^T)^-1]_plane` of every step. The free energy along the plane
offset is the integral of the mean multiplier minus `kT ln <Z^(-1/2)>`. A plane
normal that translates rigid groups as a whole is unaffected.
