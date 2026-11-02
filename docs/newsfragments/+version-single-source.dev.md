The release version is written by hand only in the root `Cargo.toml`.
The key is `version` in `[workspace.package]`.
`rgpot-core` and `rgpot-pair` inherit it.
cogapp regenerates every other copy from it.
Copies sit in the meson, CMake, pyproject, pixi and towncrier files.
The C header and the rgpot-core README carry copies as well.
A prek hook fails a pull request when a copy drifts.
