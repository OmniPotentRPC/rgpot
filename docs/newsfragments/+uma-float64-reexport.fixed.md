A UMA package exported before the edge-range check is refused, so it
cannot be scored in float64. The package is AOTI (AOTInductor,
PyTorch ahead-of-time compiled). `scripts/reexport_uma_float64.sh`
retraces each one from its weights and the original export config,
records `pos_dtype`, `model_dtype` and the traced edge count, and
writes a receipt of the input and output sha256 and the dtype.
