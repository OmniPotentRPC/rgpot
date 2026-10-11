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

# F1: FREE_THREADED sets Py_mod_gil = Py_MOD_GIL_NOT_USED on a
# free-threaded interpreter. On a GIL interpreter nanobind keeps
# STABLE_ABI and ignores FREE_THREADED. cp312-abi3 covers 3.12+ GIL.
# cp314t and cp315t are version-specific free-threaded wheels.
nanobind_add_module(_core FREE_THREADED STABLE_ABI NB_STATIC src/ext.cpp)
target_link_libraries(_core PRIVATE rust_core_demo)

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

nanobind_add_stub(
  _core_stub
  MODULE _core
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_core.pyi"
  PYTHON_PATH $<TARGET_FILE_DIR:_core>
  DEPENDS _core
)

install(TARGETS _core LIBRARY DESTINATION .)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_core.pyi" DESTINATION .)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/py.typed" DESTINATION .)
