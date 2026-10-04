# The same kernels and workspace back every C++ Fortran potential.
find_package(Threads REQUIRED)
include(CheckFortranCompilerFlag)
set(_rgpot_fortran_options)
foreach(flag IN ITEMS -std=f2018 -fimplicit-none -Wall
                      -Wno-unused-dummy-argument -fvisibility=hidden)
  string(MAKE_C_IDENTIFIER "RGPOT_FORTRAN_${flag}" supported)
  check_fortran_compiler_flag("${flag}" ${supported})
  if(${supported})
    list(APPEND _rgpot_fortran_options "${flag}")
  endif()
endforeach()

set(_rgpot_vesin_fortran_sources
    "${PROJECT_SOURCE_DIR}/third_party/vesin/fortran/cdef.f90"
    "${PROJECT_SOURCE_DIR}/third_party/vesin/fortran/vesin.f90")
set(_rgpot_common_sources
    rgpot_kinds.f90 rgpot_ferror.f90 rgpot_neighbors.f90 rgpot_workspace.f90)
set(_rgpot_kernel_sources
    rgpot_cuh2.f90 rgpot_cuh2_capi.f90
    rgpot_eam_al.f90 rgpot_eam_al_capi.f90
    rgpot_edip.f90 rgpot_edip_capi.f90
    rgpot_fehe.f90 rgpot_fehe_capi.f90
    rgpot_lenosky.f90 rgpot_lenosky_capi.f90
    rgpot_sw.f90 rgpot_sw_capi.f90
    rgpot_tersoff.f90 rgpot_tersoff_capi.f90
    rgpot_water_h.f90 rgpot_water_h_capi.f90)

function(rgpot_fortran_target target module_directory hide_symbols)
  file(MAKE_DIRECTORY "${module_directory}")
  set_target_properties(${target} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    Fortran_MODULE_DIRECTORY "${module_directory}")
  target_include_directories(${target}
    PRIVATE "${PROJECT_SOURCE_DIR}/third_party/vesin"
    PUBLIC "$<BUILD_INTERFACE:${module_directory}>")
  target_link_libraries(${target} PRIVATE rgpot_vesin)
  foreach(flag IN LISTS _rgpot_fortran_options)
    if(NOT hide_symbols AND flag STREQUAL "-fvisibility=hidden")
      continue()
    endif()
    target_compile_options(${target} PRIVATE
      "$<$<COMPILE_LANGUAGE:Fortran>:${flag}>")
  endforeach()
endfunction()

# Keeping the kernels in an archive lets librgpot hide their module and
# bind(c) symbols while leaving the C++ potential classes visible.
add_library(rgpot_fortran_kernels STATIC
  ${_rgpot_vesin_fortran_sources} ${_rgpot_common_sources} ${_rgpot_kernel_sources})
rgpot_fortran_target(rgpot_fortran_kernels "${CMAKE_CURRENT_BINARY_DIR}/modules" TRUE)
set_target_properties(rgpot_fortran_kernels PROPERTIES CXX_VISIBILITY_PRESET hidden)

# A C++-only installed consumer does not initialize CMake's Fortran
# runtime discovery. Carry the additional runtime libraries in the
# archive's link interface without requiring a Fortran compiler there.
set(_rgpot_fortran_runtime ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
list(REMOVE_ITEM _rgpot_fortran_runtime ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES})
list(REMOVE_DUPLICATES _rgpot_fortran_runtime)
foreach(runtime IN LISTS _rgpot_fortran_runtime)
  target_link_libraries(rgpot_fortran_kernels PRIVATE
    "$<INSTALL_INTERFACE:${runtime}>")
endforeach()

# The direct CuH2 plugin is a separate public bind(c) entry point.
add_library(rgpot_cuh2 SHARED ${_rgpot_vesin_fortran_sources} ${_rgpot_common_sources}
  rgpot_cuh2.f90 rgpot_cuh2_capi.f90)
rgpot_fortran_target(rgpot_cuh2 "${CMAKE_CURRENT_BINARY_DIR}/cuh2-modules" FALSE)

if(RGPOT_BUILD_TESTS)
  function(rgpot_fortran_test name source timeout)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE rgpot_fortran_kernels)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES TIMEOUT ${timeout})
  endfunction()
  rgpot_fortran_test(FortranSWGather test_sw_gather.f90 120)
  rgpot_fortran_test(FortranCuH2Gather test_cuh2.f90 120)
  rgpot_fortran_test(FortranEAMAlGather test_eam_al.f90 120)
  rgpot_fortran_test(FortranWaterHGather test_water_h.f90 120)
  rgpot_fortran_test(FortranFeHeGather test_fehe.f90 120)
  rgpot_fortran_test(FortranTersoffGather test_tersoff.f90 180)
  rgpot_fortran_test(FortranEDIPGather test_edip.f90 180)
  rgpot_fortran_test(FortranLenoskyGather test_lenosky.f90 180)
endif()
