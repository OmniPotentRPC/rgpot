eam_openmp = dependency('openmp', language: 'cpp', required: false)
eam_deps = _deps
if eam_openmp.found()
    eam_deps += eam_openmp
    # Later link lines copy _deps after this subdir. The umbrella library
    # keeps the earlier snapshot in _deps_sans_core and adds eam_openmp itself.
    _deps += eam_openmp
endif
# The streaming walk in EAMFixed.cc is sensitive to where the linker
# places it. A shorter object ahead of this archive moved the walk onto
# a slower boundary, and the 256-atom one-thread stream went from under
# develop to entirely above it. Aligning the functions in this file
# keeps that walk on a fast boundary.
eam_fixed = static_library(
    'eam_fixed_aligned',
    ['EAMFixed.cc'],
    dependencies: eam_deps,
    cpp_args: _args + _fp_contract_off + ['-DEAM_STANDALONE', '-falign-functions=64'],
    include_directories: ['../../', '.'],
    pic: true,
    install: false,
)
eam_cell = static_library(
    'eam_cell',
    ['EAM.cpp', 'EAMPot.cc'],
    objects: eam_fixed.extract_all_objects(recursive: false),
    dependencies: eam_deps,
    cpp_args: _args + _fp_contract_off + ['-DEAM_STANDALONE'],
    include_directories: ['../../', '.'],
    pic: true,
    install: false,
)
