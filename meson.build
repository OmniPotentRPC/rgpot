project(
    'rgpot',
    'cpp',
    version: '3.0.2',
    default_options: ['warning_level=2', 'cpp_std=c++20'],
)
# IMPORTANT!! warning_level=3 passes -fimplicit-none
# Many of the older Fortran codes need implicit typing

host_system = host_machine.system()

# Add C++ compiler options
_cppc_early = meson.get_compiler('cpp')
if _cppc_early.get_id() == 'msvc' or _cppc_early.get_id() == 'clang-cl'
    # WIN32_LEAN_AND_MEAN / NOMINMAX: keep windows.h from defining COM
    # symbols (e.g. IPrintDialogServices) that break Cap'n Proto templates
    # under MSVC when any TU mixes windows.h with capnp headers.
    cpp_args = [
        '/W3',
        '/DWIN32_LEAN_AND_MEAN',
        '/DNOMINMAX',
    ]
else
    cpp_args = ['-Wall', '-Wextra', '-Wpedantic']
endif
_args = [cpp_args]  # Extra arguments
_deps = []  # Dependencies
_linkto = []  # All the sub-libraries (potentials)
_incdirs = []  # All the includes
_rgpot_srcs = []

# Add conditionals
if host_system == 'darwin'
    _args += ['-DOSX=TRUE']
    cpp_args += ['-faligned-allocation']
endif

# Languages
add_languages('c', required: true)
# Fortran pot kernels on all hosts where a compiler is available. On
# Darwin/CI, pair with system clang++ (not conda arm64-apple-darwin20 clang)
# so C++ targets link the host SDK.
#
# with_fortran_pots is independent of the RPC options: client-only builds
# skip the main library (and with it every pot), but the decision belongs
# to the pot option, not to the RPC role.
_fortran_pots_opt = get_option('with_fortran_pots')
if _fortran_pots_opt.disabled()
    has_fortran = false
else
    has_fortran = add_languages(
        'fortran',
        required: _fortran_pots_opt.enabled(),
    )
endif
fc = disabler()
if has_fortran
    fc = meson.get_compiler('fortran')
    # Matches the RGPOT_HAS_FORTRAN_POTS cflag rgpot.pc exports, so in-tree
    # sources and installed consumers gate the Fortran pots on one name.
    _args += ['-DRGPOT_HAS_FORTRAN_POTS=TRUE']
endif

rpc_enabled = get_option('with_rpc') or get_option('with_rpc_client_only')

if get_option('with_rpc_client_only')
    _args += ['-DRGPOT_RPC_CLIENT_ONLY=TRUE']
endif

cc = meson.get_compiler('c')
cppc = meson.get_compiler('cpp')

# Dependencies
# libm for Unix systems
m_dep = cppc.find_library('m', required: false)
_deps += m_dep
# libstdc++ only on Linux/GCC; Darwin/clang uses libc++ via the driver.
if host_system == 'linux'
    _deps += [declare_dependency(link_args: '-lstdc++')]
endif

# ------------------------ vesin (neighbour lists)
# One vesin for the whole tree: the nlist headers, MetatomicPot, and the
# Fortran pots all consume this dependency. Prefer an installed vesin
# >= 0.6; fall back to the vendored single-TU build, which carries the
# local patches maintained as vesin upstream candidates.
vesin_threads_dep = dependency('threads')
vesin_dep = dependency('vesin', version: '>=0.6', required: false)
# vesin_hdr_dep carries compile information only. Meson copies the objects
# of a static library into every static library that links it, so the
# convenience archives that link_whole into librgpot must see headers
# alone; the objects enter the umbrella once through vesin_dep.
vesin_hdr_dep = vesin_dep
if not vesin_dep.found()
    vesin_int = static_library(
        'vesin_internal',
        ['third_party/vesin/vesin-single-build.cpp'],
        dependencies: [vesin_threads_dep],
        # Consumers link statically; hidden symbols stay out of the
        # enclosing shared objects' dynamic tables.
        cpp_args: cppc.get_supported_arguments(['-fvisibility=hidden']),
        pic: true,
    )
    vesin_hdr_dep = declare_dependency(
        dependencies: [vesin_threads_dep],
        include_directories: include_directories('third_party/vesin'),
    )
    vesin_dep = declare_dependency(
        link_with: vesin_int,
        dependencies: [vesin_hdr_dep],
    )
    message('vesin: vendored third_party/vesin')
else
    message('vesin: system / pkg-config')
endif
_deps += vesin_hdr_dep

# vesin Fortran interface, vendored at the same version. Fortran pots take
# `use vesin` through vesin_f_dep (.mod directory propagated).
vesin_f_dep = disabler()
if has_fortran
    vesin_fortran_lib = static_library(
        'vesin_fortran',
        [
            'third_party/vesin/fortran/cdef.f90',
            'third_party/vesin/fortran/vesin.f90',
        ],
        dependencies: vesin_hdr_dep,
        pic: true,
    )
    vesin_f_dep = declare_dependency(
        link_with: vesin_fortran_lib,
        dependencies: vesin_hdr_dep,
        include_directories: vesin_fortran_lib.private_dir_include(),
    )
endif

# Fortran runtime for the umbrella. librgpot is a C++ target that takes the
# kernels through link_whole, which leaves meson no Fortran source of its
# own to infer the runtime from.
#
# Windows + MSVC + Flang: flang-new embeds /DEFAULTLIB:FortranRuntime.dynamic.lib
# (and FortranDecimal.dynamic.lib) in COFF objects. Those libs live next to the
# compiler under Library/lib on conda-forge / pixi. Search that dir so link.exe
# gets full paths rather than LNK1104 on a bare DEFAULTLIB name.
fortran_rt_deps = []
if has_fortran
    _fc = meson.get_compiler('fortran')
    _fc_id = _fc.get_id()
    _fc_rt_names = []
    _fc_rt_dirs = []
    if _fc_id == 'gcc'
        _fc_rt_names = ['gfortran']
    elif _fc_id == 'llvm-flang' or _fc_id == 'flang'
        if host_system == 'windows'
            _fc_rt_names = [
                'FortranRuntime.dynamic',
                'FortranDecimal.dynamic',
                'FortranRuntime.static',
                'FortranDecimal.static',
                'FortranRuntime',
                'FortranDecimal',
                'flang_rt.runtime',
                'flang',
            ]
            _fs = import('fs')
            _fc_bin = _fc.cmd_array()[0]
            if _fs.is_absolute(_fc_bin) or _fs.exists(_fc_bin)
                _fc_bindir = _fs.parent(_fc_bin)
                # .../Library/bin -> .../Library/lib
                _fc_rt_dirs += [_fs.parent(_fc_bindir) / 'lib']
                # .../bin -> .../lib (some layouts)
                _fc_rt_dirs += [_fc_bindir / '..' / 'lib']
            endif
        else
            _fc_rt_names = ['flang_rt.runtime', 'FortranRuntime', 'flang']
        endif
    elif _fc_id == 'intel-llvm' or _fc_id == 'intel'
        _fc_rt_names = ['ifcoremt']
    endif
    foreach _n : _fc_rt_names
        if _fc_rt_dirs.length() > 0
            _rt = _fc.find_library(_n, dirs: _fc_rt_dirs, required: false)
        else
            _rt = _fc.find_library(_n, required: false)
        endif
        if _rt.found()
            fortran_rt_deps += _rt
            message('fortran runtime: ' + _n)
            if host_system != 'windows'
                break
            endif
        endif
    endforeach
endif

# --------------------- Optional dependencies
# Each dependency is controlled by a single flag; examples select
# themselves based on which deps are available (see CppCore/meson.build).

has_eigen = false
if get_option('with_eigen') or (
    get_option('with_examples')
    and not get_option(
    'with_rpc_client_only',
    )
)
    eigen_dep = dependency('eigen3', required: false)
    if not eigen_dep.found()
        eigen_dep = dependency(
            'eigen3',
            subproject: subproject('eigen'),
            required: get_option('with_eigen'),
        )
    endif
    if eigen_dep.found()
        _deps += eigen_dep
        has_eigen = true
    endif
endif

has_xtensor = false
if get_option('with_xtensor')
    _deps += dependency('xtensor')
    _deps += dependency('xtensor-blas')
    _args += ['-DWITH_XTENSOR=TRUE']
    has_xtensor = true
endif

has_fmt = false
if not get_option('pure_lib') or (
    get_option('with_examples')
    and not get_option(
    'with_rpc_client_only',
    )
)
    _args += ['-DNOT_PURE_LIB=TRUE']
    fmt_dep = dependency('fmt', required: true)
    _deps += fmt_dep
    has_fmt = true
endif

has_xtb = false
if get_option('with_xtb')
    xtb_dep = dependency('xtb', required: true)
    _deps += xtb_dep
    _args += ['-DRGPOT_HAS_XTB=TRUE']
    has_xtb = true
endif

has_xckernel = false
if get_option('with_xckernel')
    _args += ['-DRGPOT_HAS_XCKERNEL=TRUE']
    has_xckernel = true
endif

has_dftd3 = false
if get_option('with_dftd3')
    dftd3_dep = dependency('s-dftd3', required: true)
    _deps += dftd3_dep
    _args += ['-DRGPOT_HAS_DFTD3=TRUE']
    has_dftd3 = true
endif

has_dftd4 = false
if get_option('with_dftd4')
    dftd4_dep = dependency('dftd4', required: true)
    _deps += dftd4_dep
    _args += ['-DRGPOT_HAS_DFTD4=TRUE']
    has_dftd4 = true
endif

# NWChemPot frontend always builds and dlopens the split nwchemc engine
# (libnwchemc.so) at runtime. rgpot carries no in-process NWChem embed.
_args += ['-DRGPOT_HAS_NWCHEM_FRONTEND=TRUE']

has_tblite = false
if get_option('with_tblite')
    tblite_dep = dependency('tblite', required: true)
    _deps += tblite_dep
    _args += ['-DRGPOT_HAS_TBLITE=TRUE']
    has_tblite = true
endif

has_metatomic = false
if get_option('with_metatomic')
    # Discover torch without importing it (import can load NumPy C extensions
    # during meson configure). Supports multi-ABI builds via:
    #   RGPOT_TORCH_ROOT=/path/to/site-packages/torch  (isolated major)
    #   or first torch/version.py on sys.path (PYTHONPATH prefixes).
    # Prefer manual -isystem/-L over cmake Torch: old/CUDA wheels break cmake
    # with missing nvcc; libtorch_cpu.so is enough for the engine plugin.
    # Metatomic multi-ABI headers always come from the host purelib.
    _py_probe = run_command(
        find_program('python3'),
        '-c',
        '''
import os
import re
import sys
import sysconfig
from pathlib import Path

purelib = Path(sysconfig.get_paths()["purelib"])

def torch_root() -> Path:
    env = os.environ.get("RGPOT_TORCH_ROOT", "").strip()
    if env:
        p = Path(env)
        if (p / "version.py").is_file():
            return p
        if (p / "torch" / "version.py").is_file():
            return p / "torch"
        raise SystemExit(f"RGPOT_TORCH_ROOT has no version.py: {env}")
    for entry in sys.path:
        cand = Path(entry) / "torch" / "version.py"
        if cand.is_file():
            return cand.parent
    host = purelib / "torch" / "version.py"
    if host.is_file():
        return host.parent
    raise SystemExit("torch not found (set RGPOT_TORCH_ROOT or install torch)")

troot = torch_root()
version_file = troot / "version.py"
match = re.search(
    r"^__version__\s*=\s*['\"]([^'\"]+)",
    version_file.read_text(),
    re.MULTILINE,
)
if match is None:
    raise SystemExit(f"torch version not found in {version_file}")
version = ".".join(match.group(1).split("+")[0].split(".")[:2])
libdir = troot / "lib"
inc = troot / "include"
api_inc = troot / "include" / "torch" / "csrc" / "api" / "include"
if not (libdir / "libtorch_cpu.so").is_file() and not (libdir / "libtorch_cpu.dylib").is_file():
    raise SystemExit(f"libtorch_cpu not under {libdir}")
print(purelib)
print(version)
print(libdir)
print(inc)
print(api_inc)
''',
        check: true,
    ).stdout().strip().split('\n')
    _py_purelib = _py_probe[0]
    _torch_major = _py_probe[1]
    _torch_libdir = _py_probe[2]
    _torch_inc = _py_probe[3]
    _torch_api_inc = _py_probe[4]
    # Manual torch dep (no cmake): includes + link search paths only.
    # Portable RUNPATH is install_rpath / repair script.
    #
    # -rpath-link and --as-needed are GNU ld only; Apple's ld rejects them
    # outright ("unknown options"), and it resolves the dylibs from -L plus
    # their install_name anyway. Older libtorch splits symbols across
    # libtorch + libtorch_cpu + libc10, which is what --no-as-needed guards
    # on the GNU side.
    _torch_link_args = ['-L' + _torch_libdir]
    if host_system == 'darwin'
        _torch_link_args += ['-ltorch', '-ltorch_cpu', '-lc10']
    else
        _torch_link_args += [
            '-Wl,-rpath-link,' + _torch_libdir,
            '-Wl,--no-as-needed',
            '-ltorch',
            '-ltorch_cpu',
            '-lc10',
            '-Wl,--as-needed',
        ]
    endif
    torch_dep = declare_dependency(
        compile_args: [
            '-isystem',
            _torch_inc,
            '-isystem',
            _torch_api_inc,
        ],
        link_args: _torch_link_args,
    )
    _deps += torch_dep

    # metatensor-torch / metatomic-torch headers live in the
    # python interpreter's site-packages under
    # <purelib>/metatensor/torch/torch-X.Y/include and
    # <purelib>/metatomic/torch/torch-X.Y/include. The cmake /
    # pkg-config dependency lookups don't see those, and
    # find_library() doesn't add include paths. Ask sysconfig for
    # the right purelib and propagate -isystem includes manually.
    # Layout (current PyPI wheels):
    #   metatensor/lib/libmetatensor.so
    #   metatensor_torch/torch-X.Y/{lib,include}
    #   metatomic/torch/torch-X.Y/{lib,include}
    #   vesin/lib/libvesin.so
    _mts_torch_root = _py_purelib / 'metatensor_torch' / (
        'torch-' + _torch_major
    )
    _mta_torch_root = _py_purelib / 'metatomic' / 'torch' / (
        'torch-' + _torch_major
    )
    # Legacy layout fallback
    _mts_torch_legacy = _py_purelib / 'metatensor' / 'torch' / (
        'torch-' + _torch_major
    )

    _mts_inc_dirs = [
        _py_purelib / 'metatensor' / 'include',
        _mts_torch_root / 'include',
        _mts_torch_legacy / 'include',
        _mta_torch_root / 'include',
        _py_purelib / 'vesin' / 'include',
    ]
    _mts_inc_args = []
    foreach d : _mts_inc_dirs
        _mts_inc_args += ['-isystem', d]
    endforeach

    # Always use purelib find_library + -isystem (no pkg-config/cmake): multi-ABI
    # builds must not pull host Torch cmake includes (dtype symbol conflicts).
    _mts_lib_dirs = [
        _mts_torch_root / 'lib',
        _mts_torch_legacy / 'lib',
        _py_purelib / 'metatensor' / 'lib',
    ]
    mts_dep = declare_dependency(
        dependencies: [
            cppc.find_library(
                'metatensor',
                dirs: [_py_purelib / 'metatensor' / 'lib'],
            ),
            cppc.find_library(
                'metatensor_torch',
                dirs: _mts_lib_dirs,
            ),
        ],
        compile_args: _mts_inc_args,
    )
    _deps += mts_dep

    mta_dep = cppc.find_library(
        'metatomic_torch',
        dirs: [
            _mta_torch_root / 'lib',
        ],
    )
    _deps += mta_dep
    # Link search paths only (-L); portable RUNPATH is install_rpath / repair.
    _deps += declare_dependency(
        link_args: [
            '-L' + (_mts_torch_root / 'lib'),
            '-L' + (_mta_torch_root / 'lib'),
            '-L' + (_py_purelib / 'metatensor' / 'lib'),
        ],
    )

    # vesin comes from the top-level dependency (vendored static fallback):
    # compile-time and runtime layouts always match, so the purelib lookup
    # is unnecessary here.

    _args += ['-DRGPOT_HAS_METATOMIC=TRUE']
    has_metatomic = true
endif

# --------------------- Rust Core (rgpot-core)
#
# Build the Rust crate via Cargo and link the resulting static library.
# The crate also generates include/rgpot.h via cbindgen.

if get_option('with_rust_core')
    cargo = find_program('cargo', required: true)

    # Determine Cargo build profile and library name
    if get_option('buildtype') == 'release'
        _cargo_profile_dir = 'release'
        _cargo_args = ['build', '--release']
    else
        _cargo_profile_dir = 'debug'
        _cargo_args = ['build']
    endif

    # Add feature flags for the Rust crate
    _cargo_features = []
    if rpc_enabled
        _cargo_features += ['rpc']
        _args += ['-DRGPOT_HAS_RPC=TRUE']
    endif
    if get_option('with_cache')
        _cargo_features += ['cache']
    endif

    if _cargo_features.length() > 0
        _cargo_args += ['--features', ','.join(_cargo_features)]
    endif

    _rgpot_core_dir = meson.current_source_dir() / 'rgpot-core'

    # Build the Rust crate; produces librgpot_core.a and include/rgpot.h
    rgpot_rust_target = custom_target(
        'rgpot_rust_core',
        command: [
            cargo,
            _cargo_args,
            '--manifest-path',
            _rgpot_core_dir / 'Cargo.toml',
            '--target-dir',
            meson.current_build_dir() / 'cargo-target',
        ],
        output: ['librgpot_core.a'],
        build_by_default: true,
        console: true,
    )

    # Link the Rust static library
    _rust_target_dir = meson.current_build_dir() / 'cargo-target' / _cargo_profile_dir
    rgpot_rust_lib = cppc.find_library(
        'rgpot_core',
        dirs: [_rust_target_dir],
        required: true,
        static: true,
    )

    # Thread and dl libraries needed by the Rust runtime
    thread_dep = dependency('threads')
    dl_dep = cppc.find_library('dl', required: false)

    _deps += [rgpot_rust_lib, thread_dep, dl_dep]

    # Include paths: generated C header + C++ RAII wrappers
    _incdirs += [
        include_directories('rgpot-core' / 'include'),  # rgpot.h (cbindgen)
        include_directories('include'),  # rgpot/*.hpp (C++ RAII)
    ]
endif

# --------------------- Projects
_incdirs += [include_directories('CppCore')]
subdir('CppCore')
rgpot_dep = declare_dependency(
    include_directories: _incdirs,
    link_with: _linkto,
    compile_args: _args,
    dependencies: _deps,
)

# --------------------- Python package (pip install rgpot)
if get_option('with_python')
    if get_option('with_rpc_client_only')
        error(
            'with_python requires the full pot library (not with_rpc_client_only)',
        )
    endif
    subdir('python')
endif
