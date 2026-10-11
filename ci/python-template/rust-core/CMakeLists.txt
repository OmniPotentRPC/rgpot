cmake_minimum_required(VERSION 3.15...3.30)
project(rust_core_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(MSVC)
  add_compile_options(/fp:strict)
else()
  add_compile_options(-ffp-contract=off)
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# T2: nightly rustc, an explicit target triple, and a std rebuilt with
# the thread sanitizer. Rust_TOOLCHAIN and Rust_CARGO_TARGET are read
# when Corrosion is included, so they are set first. The workflow passes
# the dated nightly that matches the image clang.
option(RUST_CORE_TSAN "Build the Rust staticlib with -Zsanitizer=thread (T2)." OFF)
if(RUST_CORE_TSAN)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(_rust_core_tsan_target "aarch64-unknown-linux-gnu")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(_rust_core_tsan_target "x86_64-unknown-linux-gnu")
  else()
    message(FATAL_ERROR
      "RUST_CORE_TSAN has no target triple for ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  if(NOT Rust_TOOLCHAIN)
    set(Rust_TOOLCHAIN "nightly" CACHE STRING
      "Rustup toolchain used when RUST_CORE_TSAN is ON")
  endif()
  set(Rust_CARGO_TARGET "${_rust_core_tsan_target}" CACHE STRING
    "Explicit cargo target triple for the sanitizer staticlib" FORCE)
  message(STATUS "RUST_CORE_TSAN toolchain ${Rust_TOOLCHAIN} target ${_rust_core_tsan_target}")
endif()

include(FetchContent)
FetchContent_Declare(
  corrosion
  GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
  GIT_TAG v0.6.1
)
FetchContent_MakeAvailable(corrosion)

corrosion_import_crate(
  MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/examples/rust-core-demo/Cargo.toml"
  CRATES rust_core_demo
  CRATE_TYPES staticlib
)

if(RUST_CORE_TSAN)
  corrosion_set_cargo_flags(rust_core_demo "-Zbuild-std=std,panic_abort")
  corrosion_add_target_rustflags(rust_core_demo "-Zsanitizer=thread")
endif()

# F1: FREE_THREADED sets Py_mod_gil = Py_MOD_GIL_NOT_USED on a
# free-threaded interpreter. On a GIL interpreter nanobind keeps
# STABLE_ABI and ignores FREE_THREADED. cp312-abi3 covers 3.12+ GIL.
# cp314t and cp315t are version-specific free-threaded wheels.
nanobind_add_module(_core FREE_THREADED STABLE_ABI NB_STATIC src/ext.cpp)
target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(_core PRIVATE rust_core_demo)

# The stub generator imports rust_core_demo._core, so the extension has
# to sit inside a package directory on PYTHON_PATH.
set(_core_stage "${CMAKE_CURRENT_BINARY_DIR}/rust_core_demo")
file(MAKE_DIRECTORY "${_core_stage}")
file(WRITE "${_core_stage}/__init__.py" "")
# A generator expression stops the Visual Studio generator from appending
# a per-configuration directory. Stubgen imports rust_core_demo._core from
# this directory, so the extension cannot land in Release/.
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "$<1:${_core_stage}>"
  RUNTIME_OUTPUT_DIRECTORY "$<1:${_core_stage}>")

if(NOT MSVC)
  target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# F9: Windows free-threaded builds define Py_GIL_DISABLED when the
# Python headers did not.
if(WIN32 AND NB_FREE_THREADED)
  target_compile_definitions(_core PRIVATE Py_GIL_DISABLED=1)
endif()

# F9: hidden visibility is the default above. --exclude-libs,ALL drops
# the staticlib from the dynamic table. The version script (Linux) and
# the export list (macOS) leave only PyInit__core.
if(UNIX AND NOT APPLE)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/core_exports.map"
"{\n  global:\n    PyInit__core;\n  local:\n    *;\n};\n")
  target_link_options(_core PRIVATE
    "LINKER:--exclude-libs,ALL"
    "LINKER:--version-script,${CMAKE_CURRENT_BINARY_DIR}/core_exports.map")
endif()
if(APPLE)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/core_exports.txt" "_PyInit__core\n")
  target_link_options(_core PRIVATE
    "LINKER:-exported_symbols_list,${CMAKE_CURRENT_BINARY_DIR}/core_exports.txt")
endif()
# A Rust staticlib exports every public symbol. On Windows the shared
# library limits that set with a module-definition file (F9).
# __declspec(dllexport) symbols from the static nanobind library are not
# removed by the .def file, so the export directory is rewritten after
# the link.
if(MSVC)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/core_exports.def" "EXPORTS\n  PyInit__core\n")
  target_link_options(_core PRIVATE
    "/DEF:${CMAKE_CURRENT_BINARY_DIR}/core_exports.def")
  add_custom_command(
    TARGET _core POST_BUILD
    COMMAND "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/ci/limit_pe_exports.py"
            "$<TARGET_FILE:_core>"
            "PyInit__core"
    VERBATIM)
endif()

nanobind_add_stub(
  _core_stub
  MODULE rust_core_demo._core
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_core.pyi"
  PYTHON_PATH "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS _core
)

# LIBRARY is the .so on Unix. RUNTIME is the .pyd on Windows.
install(TARGETS _core
  LIBRARY DESTINATION rust_core_demo
  RUNTIME DESTINATION rust_core_demo)
