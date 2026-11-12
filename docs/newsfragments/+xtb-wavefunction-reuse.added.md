``XTBPot`` and ``XTBDlopen`` reuse the SCC wavefunction across calls.
Reuse holds while every atom stays within ``warm_move`` angstroms.
The default guard is 1 angstrom.
``reset()`` drops that guess, as does a larger move or a new composition.
The next single point then uses xtb's own guess.
The engine exports ``rgpot_xtb_reset`` and ``rgpot_xtb_set_warm_move``.
It also exports ``rgpot_xtb_warm_move`` and ``rgpot_xtb_reused_wavefunction``.
