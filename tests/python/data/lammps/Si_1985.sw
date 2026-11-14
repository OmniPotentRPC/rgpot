# Stillinger-Weber silicon, F. H. Stillinger and T. A. Weber,
# Phys. Rev. B 31, 5262 (1985), in the LAMMPS pair_style sw format.
# These are the constants of rgpot::StillingerWeberSi: epsilon is the
# published 3.4723e-12 erg in eV (CODATA electron-volt) and costheta0 is
# the double nearest -1/3. The LAMMPS potentials/Si.sw file rounds epsilon
# to 2.1683 eV and costheta0 to -0.333333333333.
# tests/python/test_lammps_agreement.py checks every value against
# StillingerWeberSi.params().
#
# element1 element2 element3 epsilon sigma a lambda gamma costheta0
#   A B p q tol
Si Si Si 2.167239195925011 2.0951 1.8 21.0 1.2 -0.3333333333333333
         7.049556277 0.6022245584 4.0 0.0 0.0
