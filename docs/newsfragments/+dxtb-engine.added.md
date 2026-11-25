Added a persistent dxtb 0.4.0 GFN1/GFN2-xTB engine. It serves float64
energy, force, and exact Torch double-backward Hessian-vector products through
the generic engine ABI. The Python executable comes from
`BuiltinParams.library`, then `RGPOT_DXTB_PYTHON`, then `python3`, and the
methods may be spelled `gfn1`, `gfn1-xtb`, `gfn2` or `gfn2-xtb`.
`rgpot_engine_available` returns 0 when that Python cannot import Torch, dxtb
0.4.0 and tad-libcint 0.1.2 or newer, and caches the verdict per executable;
creation with such a Python fails with a message naming the executable and what
it lacks. The worker is started with `posix_spawnp`, so creation is safe in a
multithreaded host, and starts when descriptors 0 and 1 are free; its socket is
closed on exec. A charge or spin the engine cannot take returns code 3 and
leaves the held configuration unchanged. A worker that exits, or sends a reply
with a bad header or larger than the request can produce, fails the call with
return code -1 and is dropped without reading the rest. Destroying the engine
lets the worker finish shutdown, treats a broken pipe during shutdown as a clean
close, and kills and reaps a worker that does not exit within a bounded time.
