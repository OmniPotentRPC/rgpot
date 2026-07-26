A wheels workflow builds and publishes the Python package. rgpot had none:
2.5.4 reached PyPI hand-built from a branch, with no tag and no
main-branch source behind it, which is how the version surfaces drifted
apart in the first place. A `v*` tag now builds manylinux and macOS
wheels plus an sdist with cibuildwheel and publishes them; a manual run
defaults to TestPyPI so an upload can be rehearsed before the
irreversible one.

The build step refuses a wheel whose `librgpot` carries fewer than the
eight Fortran `bind(c)` entry points, so the potentials cannot silently
fall out of the artifact again.
