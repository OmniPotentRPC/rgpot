eam_openmp = dependency('openmp', language: 'cpp', required: false)
eam_deps = _deps
if eam_openmp.found()
    eam_deps += eam_openmp
    # Later link lines copy _deps after this subdir. The umbrella library
    # keeps the earlier snapshot in _deps_sans_core and adds eam_openmp itself.
    _deps += eam_openmp
endif
eam_cell = static_library(
    'eam_cell',
    ['EAM.cpp', 'EAMPot.cc', 'EAMFixed.cc'],
    dependencies: eam_deps,
    cpp_args: _args + _fp_contract_off + ['-DEAM_STANDALONE'],
    include_directories: ['../../', '.'],
    pic: true,
    install: false,
)
