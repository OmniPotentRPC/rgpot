Repeated loads of the same engine plugin no longer accumulate dynamic-linker handles. The library stays mapped for the life of the process, and a later load of that path reuses the first handle.
