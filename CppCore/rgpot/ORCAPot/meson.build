# Copyright 2023--present rgpot developers
#
# ORCAPot: pure consumer of the split orcac library. The frontend always
# builds and resolves liborcac by dlopen; orcac.h is orcac's header,
# vendored and pinned by orcac.h.pin. rgpot carries no ORCA code.
#
# ORCALegacyPot, the deprecated runner that starts the orca executable as a
# child process, builds only with with_orca_legacy.

orca_inc = include_directories('.')

orcac_dl_dep = dependency('dl', required: false)
if not orcac_dl_dep.found()
    orcac_dl_dep = meson.get_compiler('cpp').find_library('dl', required: false)
endif

orcapot = static_library(
    'orcapot',
    'ORCAGuards.cc',
    'ORCAPot.cc',
    # Same reason as mopacpot: this archive must not absorb the core objects
    # the umbrella already link_wholes.
    dependencies: _deps_sans_core + [orcac_dl_dep],
    cpp_args: _args,
    include_directories: _incdirs + [orca_inc],
    pic: true,
    install: false,
)

orcapot_dep = declare_dependency(
    link_with: [orcapot],
    dependencies: [orcac_dl_dep],
    include_directories: _incdirs + [orca_inc],
)

if has_orca_legacy
    # Static convenience lib; ships inside the librgpot umbrella (link_whole).
    # The orca executable is found at run time: nothing here links or loads
    # ORCA.
    orca_legacy_pot = static_library(
        'orca_legacy_pot',
        'ORCAFiles.cc',
        'ORCALegacyPot.cc',
        dependencies: _deps,
        cpp_args: _args,
        include_directories: ['../../'],
        pic: true,
        install: false,
    )
endif
