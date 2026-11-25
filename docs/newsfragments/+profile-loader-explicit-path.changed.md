`rgpot::abi::ProfileLoader` now tries only the library a non-empty explicit
path names. When it does not open, `load()` throws a message that names the
path instead of falling back to the `<PREFIX>_LIBRARY` and
`RGPOT_<PREFIX>_ENGINE` hints and the default library names. The lammpc
`LammpsPot` constructor, `rgpot_lammpc_potential()` and the Python
`rgpot_lammps_potential(config_bytes, engine_path)` overload inherit this.
The lammpc constructor also refuses truncated bytes and a `PotentialConfig`
without the `lammps` arm with `std::invalid_argument` (`ValueError` in Python).
