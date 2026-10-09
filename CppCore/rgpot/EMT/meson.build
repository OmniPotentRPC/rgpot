# Vendored Asap effective-medium sources. Short includes (Atoms.h, EMT.h)
# resolve in this directory.
asap_emt = static_library(
    'asap_emt',
    [
        'Asap/NeighborList.cpp',
        'Asap/GhostAtoms.cpp',
        'Asap/CellList.cpp',
        'Asap/Vec.cpp',
        'Asap/Atoms.cpp',
        'Asap/Exception.cpp',
        'Asap/EMTRasmussenParameterProvider.cpp',
        'Asap/EMTDefaultParameterProvider.cpp',
        'Asap/EMT.cpp',
        'Asap/stacktrace.c',
        'Asap/SuperCell.cpp',
        'EMTPot.cc',
    ],
    dependencies: _deps,
    cpp_args: _args,
    include_directories: ['../../', 'Asap'],
    pic: true,
    install: false,
)
