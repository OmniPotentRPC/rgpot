The MSVC build compiles the legacy LAMMPS engine test plugin. Its
load and unload hooks are registered from the DLL entry, because
cl.exe rejects GNU constructor attributes.
