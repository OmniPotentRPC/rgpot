cmake_minimum_required(VERSION 3.15...3.30)
project(rgbind_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

option(RGBIND_TSAN "Instrument the extension with ThreadSanitizer (F2/T2)." OFF)

if(RGBIND_TSAN)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=thread)
endif()

if(MSVC)
  add_compile_options(/fp:strict)
else()
  add_compile_options(-ffp-contract=off)
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  _rgbind_test
  FREE_THREADED
  STABLE_ABI
  NB_STATIC
  tests/src/test_module.cpp
)

target_include_directories(_rgbind_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(MSVC)
  target_compile_options(_rgbind_test PRIVATE /W3 /EHsc)
else()
  target_compile_options(_rgbind_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

# F9: a Windows free-threaded build defines Py_GIL_DISABLED when the
# Python headers did not. GIL builds leave the macro unset.
if(WIN32 AND NB_FREE_THREADED)
  target_compile_definitions(_rgbind_test PRIVATE Py_GIL_DISABLED=1)
endif()

if(UNIX AND NOT APPLE)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/rgbind_exports.map"
"{\n  global:\n    PyInit__rgbind_test;\n  local:\n    *;\n};\n")
  target_link_options(_rgbind_test PRIVATE
    "LINKER:--exclude-libs,ALL"
    "LINKER:--version-script,${CMAKE_CURRENT_BINARY_DIR}/rgbind_exports.map")
endif()
if(APPLE)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/rgbind_exports.txt" "_PyInit__rgbind_test\n")
  target_link_options(_rgbind_test PRIVATE
    "LINKER:-exported_symbols_list,${CMAKE_CURRENT_BINARY_DIR}/rgbind_exports.txt")
endif()

nanobind_add_stub(
  _rgbind_test_stub
  MODULE _rgbind_test
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_rgbind_test.pyi"
  PYTHON_PATH $<TARGET_FILE_DIR:_rgbind_test>
  DEPENDS _rgbind_test
)

install(TARGETS _rgbind_test LIBRARY DESTINATION .)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_rgbind_test.pyi" DESTINATION .)

add_executable(rgbind_schedule_test tests/src/schedule_test.cpp)
target_include_directories(rgbind_schedule_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(rgbind_schedule_test PRIVATE RGBIND_NO_PYTHON)
if(NOT MSVC)
  target_compile_options(rgbind_schedule_test PRIVATE -Wall -Wextra -Wpedantic)
endif()
