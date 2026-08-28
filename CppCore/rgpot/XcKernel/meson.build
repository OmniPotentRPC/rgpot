# In-process XcKernel over the vendored first-slice libxckernel C ABI.
# Not a Potential. Built only when -Dwith_xckernel=true.

fs = import('fs')

_xck_root = meson.project_source_root() / 'third_party' / 'libxckernel'
_xck_hdr = _xck_root / 'include' / 'xckernel.h'
if not fs.is_file(_xck_hdr)
    error(
        'with_xckernel=true needs the first-slice C package at '
        + _xck_hdr
        + '. Generate it on rg.terra: scripts/gen_libxckernel.sh',
    )
endif

_xck_src_names = [
    'xck_lda_r_o0',
    'xck_lda_r_o1',
    'xck_lda_r_o2',
    'xck_lda_ua_o1',
    'xck_lda_ua_o2',
    'xck_lda_ub_o1',
    'xck_lda_ub_o2',
    'xck_lda_st_o2_p',
    'xck_lda_st_o2_m',
    'xck_gga_r_o0',
    'xck_gga_r_o1',
    'xck_gga_r_o2',
    'xck_gga_ua_o1',
    'xck_gga_ua_o2',
    'xck_gga_ub_o1',
    'xck_gga_ub_o2',
    'xck_gga_st_o2_p',
    'xck_gga_st_o2_m',
    'xck_mgga_tau_r_o0',
    'xck_mgga_tau_r_o1',
    'xck_mgga_tau_r_o2',
    'xck_mgga_tau_ua_o1',
    'xck_mgga_tau_ua_o2',
    'xck_mgga_tau_ub_o1',
    'xck_mgga_tau_ub_o2',
    'xck_mgga_tau_st_o2_p',
    'xck_mgga_tau_st_o2_m',
]
_xck_srcs = []
foreach n : _xck_src_names
    _rel = '../../../third_party/libxckernel/src' / (n + '.cpp')
    _src = _xck_root / 'src' / (n + '.cpp')
    if not fs.is_file(_src)
        error(
            'with_xckernel=true missing generated source '
            + _src
            + '. Generate on rg.terra: scripts/gen_libxckernel.sh',
        )
    endif
    _xck_srcs += files(_rel)
endforeach

xckernel_inc = include_directories(
    '../../../third_party/libxckernel/include',
)

libxckernel = static_library(
    'xckernel',
    _xck_srcs,
    include_directories: xckernel_inc,
    # Paper C-vs-NumPy bound (~1e-16) is measured at -O2 (cbackend_validate).
    cpp_args: ['-Wno-unused-parameter', '-O2'],
    pic: true,
    install: false,
)

xckernel_dep = declare_dependency(
    link_with: [libxckernel],
    include_directories: xckernel_inc,
)

xckernelpot = static_library(
    'xckernel_api',
    'XcKernel.cc',
    dependencies: _deps + [xckernel_dep],
    cpp_args: _args,
    include_directories: _incdirs + [xckernel_inc],
    pic: true,
    install: false,
)
