``XTBPot`` and ``XTBDlopen`` can run a looser SCC while the RMS force is large.
``loose_accuracy`` selects that xtb accuracy, and ``adaptive_force`` is the RMS force in eV/Angstrom that tightens the step.
A step that falls to the threshold is evaluated again at ``accuracy``.
``rgpot_xtb_set_adaptive_scc`` and ``rgpot_xtb_last_accuracy`` are the engine calls.
Zero ``loose_accuracy`` leaves every call at ``accuracy``.
