# In-process XcKernel over the vendored first-slice libxckernel C ABI.
# Not a Potential. No NEEDED on Python; generator stays off the wheel.

xck_inc = include_directories('.')
_xck_gen_inc = include_directories('../../../third_party/libxckernel/include')

_xck_first_slice = [
    'xck_lda_r_o0',
    'xck_gga_r_o0',
    'xck_mgga_tau_r_o0',
    'xck_lda_r_o1',
    'xck_lda_ua_o1',
    'xck_lda_ub_o1',
    'xck_lda_r_o2',
    'xck_lda_ua_o2',
    'xck_lda_ub_o2',
    'xck_lda_st_o2_p',
    'xck_lda_st_o2_m',
    'xck_gga_r_o1',
    'xck_gga_ua_o1',
    'xck_gga_ub_o1',
    'xck_gga_r_o2',
    'xck_gga_ua_o2',
    'xck_gga_ub_o2',
    'xck_gga_st_o2_p',
    'xck_gga_st_o2_m',
    'xck_mgga_tau_r_o1',
    'xck_mgga_tau_ua_o1',
    'xck_mgga_tau_ub_o1',
    'xck_mgga_tau_r_o2',
    'xck_mgga_tau_ua_o2',
    'xck_mgga_tau_ub_o2',
    'xck_mgga_tau_st_o2_p',
    'xck_mgga_tau_st_o2_m',
]

fs = import('fs')
_xck_root = meson.project_source_root() / 'third_party' / 'libxckernel'
if not fs.is_file(_xck_root / 'include' / 'xckernel.h')
    error(
        'with_xckernel=true but third_party/libxckernel/include/xckernel.h is missing. Generate the first-slice C package on rg.terra with scripts/gen_libxckernel.sh',
    )
endif

_xck_src = files('XcKernel.cc')
foreach _name : _xck_first_slice
    _tu = _xck_root / 'src' / (_name + '.cpp')
    if not fs.is_file(_tu)
        error('with_xckernel=true missing generated ' + _tu)
    endif
    _xck_src += files(_tu)
endforeach

xckernel = static_library(
    'xckernel',
    _xck_src,
    dependencies: _deps,
    cpp_args: _args,
    include_directories: _incdirs + [xck_inc, _xck_gen_inc],
    pic: true,
    install: false,
)

xckernel_dep = declare_dependency(
    link_with: [xckernel],
    include_directories: _incdirs + [xck_inc, _xck_gen_inc],
)
