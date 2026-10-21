The MOPAC frontend stays inside `librgpot` and is not installed as its own archive. Installing that archive copied the core helpers into it and the shared library then failed to link.
