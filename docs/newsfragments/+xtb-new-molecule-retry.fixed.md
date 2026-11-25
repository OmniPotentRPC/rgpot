`XTBPot`, and the xtb engine behind `XTBDlopen`, now builds a new molecule
on the call after `xtb_newMolecule` refuses one. The handle used to keep
the composition of the call before the failure, so a later call with that
composition went on to update a molecule that no longer existed.
