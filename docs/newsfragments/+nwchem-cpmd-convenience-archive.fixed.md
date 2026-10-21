The NWChem and CPMD frontends stay inside ``librgpot`` and are not installed
as their own archives. Installing those archives could copy the core helpers
into them and make the shared library fail to link. A client-only build,
which has no umbrella library, still installs the archives.
