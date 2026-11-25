The dxtb engine's `rgpot_engine_available` now returns 0 when the Python
that `RGPOT_DXTB_PYTHON` or `python3` selects cannot import Torch, dxtb
0.4.0 and tad-libcint 0.1.2 or newer. It checks each executable once and
caches the verdict, so a host learns before creation that the engine cannot
run.
