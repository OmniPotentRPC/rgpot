Exported rgpot_lammpc_potential from the shared library and added it to
the ABI baseline checks so hidden-visibility and Windows builds keep the
constructor in the public C surface.
