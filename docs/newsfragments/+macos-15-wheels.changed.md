macOS wheels of the main package are built on the `macos-15` runner
and tagged `macosx_15_0`.
Homebrew on that image builds libgfortran, libquadmath, OpenBLAS and
libomp for macOS 15.0, and delocate rejects them under a 14.0 target.
The bind and rust-core template wheels stay `macosx_14_0`.
