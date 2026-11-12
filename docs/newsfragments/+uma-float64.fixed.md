UmaPot builds positions, the cell and cell offsets in the
package dtype. The package is AOTI (AOTInductor, PyTorch
ahead-of-time compiled). It records `pos_dtype`, and
`model_dtype` when `pos_dtype` is absent. The exporter
converts the model with `hydra.double()` for `--dtype
float64`, and calls `hydra.float()` for `--dtype float32`.
A package with neither key stays float32. UmaPot logs that
fallback and both dtypes at load.
