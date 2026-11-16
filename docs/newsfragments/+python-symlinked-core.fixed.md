`import rgpot` finds `librgpot` when `rgpot._core` is reached through a
symlink, such as an extension linked from a meson build tree into
`python/rgpot` or a staged copy of the package. No `LD_LIBRARY_PATH` is
needed. glibc expands the extension's `$ORIGIN` runpath from the symlink's
directory, so the import failed with `librgpot.so.6: cannot open shared
object file`, and the six `ipi_driver_ase` cases that compare against
`RgpotCalculator` skipped. The package now opens the extension from its
resolved path. On macOS the extension's runpath entries are written as
`@loader_path`, which dyld expands; `$ORIGIN` there was a literal path. On
Windows the resolved directory and `.rgpot.mesonpy.libs` join the DLL search
path.
