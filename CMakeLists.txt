cmake_minimum_required(VERSION 3.16)

project(
  rgpot
  VERSION 3.4.0
  DESCRIPTION "Cpp core for potential energy surface functionality"
  HOMEPAGE_URL "https://github.com/OmniPotentRPC/rgpot"
  LANGUAGES C CXX)

# --- Top-level detection (CMake < 3.21 compat) ---
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(RGPOT_IS_TOP_LEVEL TRUE)
else()
  set(RGPOT_IS_TOP_LEVEL FALSE)
endif()

# --- Options ---
option(RGPOT_RPC_CLIENT_ONLY "Build ONLY the RPC client bridge (lightweight)"
       OFF)

if(RGPOT_RPC_CLIENT_ONLY)
  set(CMAKE_CXX_STANDARD 17) # C++17 is sufficient for the RPC bridge
  option(RGPOT_WITH_RPC "Build with Cap'n Proto RPC support" ON)

  # Force disable heavy dependencies
  set(RGPOT_WITH_XTENSOR OFF)
  set(RGPOT_WITH_EIGEN OFF)
  set(RGPOT_WITH_CACHE OFF)
  set(RGPOT_PURE_LIB ON)
  set(RGPOT_BUILD_EXAMPLES OFF)
else()
  set(CMAKE_CXX_STANDARD 20)
  option(RGPOT_WITH_XTENSOR "Build with xtensor support" OFF)
  option(RGPOT_WITH_EIGEN "Build with Eigen3 support" OFF)
  option(RGPOT_PURE_LIB "Build as a pure library (no fmt dependency)" ON)
  option(RGPOT_WITH_CACHE "Build with RocksDB caching support" ON)
  option(RGPOT_WITH_RPC "Build with Cap'n Proto RPC support" OFF)
  option(RGPOT_BUILD_EXAMPLES "Build examples" ${RGPOT_IS_TOP_LEVEL})
endif()

option(RGPOT_BUILD_TESTS "Build tests" ${RGPOT_IS_TOP_LEVEL})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
include(FetchContent)

# 1. RPC Dependency
if(RGPOT_WITH_RPC)
  find_package(CapnProto REQUIRED)
endif()

# 1. Server/Physics Dependencies
if(NOT RGPOT_RPC_CLIENT_ONLY)
  # Core builds provide every public Fortran potential. The RPC-only
  # bridge has no numerical kernels and does not need a Fortran compiler.
  enable_language(Fortran)
  if(RGPOT_WITH_CACHE)
    # Conda macOS runners sometimes fail FindThreads before RocksDB; set the
    # pthread hint explicitly so RocksDBConfig's find_dependency(Threads) works.
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    find_package(RocksDB REQUIRED)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    find_package(xxHash QUIET)
    list(POP_BACK CMAKE_MODULE_PATH)
    if(NOT xxHash_FOUND)
      message(
        STATUS "xxHash not found via find_package, fetching from source...")
      FetchContent_Declare(
        xxHash
        GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
        GIT_TAG v0.8.3)
      FetchContent_MakeAvailable(xxHash)
      if(NOT TARGET xxHash::xxhash)
        # The pinned release keeps its CMake project below the source root.
        set(XXHASH_BUILD_XXHSUM OFF)
        set(XXHASH_BUNDLED_MODE OFF)
        add_subdirectory("${xxhash_SOURCE_DIR}/cmake_unofficial"
                         "${xxhash_BINARY_DIR}")
      endif()
      set_target_properties(xxhash PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
  endif()

  if(NOT RGPOT_PURE_LIB)
    find_package(fmt REQUIRED)
  endif()

  if(RGPOT_WITH_XTENSOR)
    find_package(xtensor REQUIRED)
    find_package(xtensor-blas REQUIRED)
  endif()

  if(RGPOT_WITH_EIGEN)
    find_package(Eigen3 3.4.0 REQUIRED)
  endif()
endif()

# --- Build Targets ---

if(RGPOT_WITH_RPC)
  set(CAPNP_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/CppCore")
  file(MAKE_DIRECTORY "${CAPNP_OUTPUT_DIR}/rgpot/rpc")

  capnp_generate_cpp(CAPNP_SOURCES CAPNP_HEADERS
                     CppCore/rgpot/rpc/Potentials.capnp)

  # On Apple, static intermediate RPC libs avoid install_name / mixed
  # toolchain dylib issues when system clang links conda Cap'n Proto; the
  # full RPC/cache feature set is still exercised in executables/tests.
  if(APPLE)
    add_library(ptlrpc STATIC ${CAPNP_SOURCES})
  else()
    add_library(ptlrpc SHARED ${CAPNP_SOURCES})
  endif()
  target_link_libraries(ptlrpc PUBLIC CapnProto::capnp-rpc)
  # Always drive C++ links with the C++ linker.
  set_target_properties(ptlrpc PROPERTIES LINKER_LANGUAGE CXX)

  # Ensure the generated headers are in the include path
  target_include_directories(
    ptlrpc
    PUBLIC $<BUILD_INTERFACE:${CAPNP_OUTPUT_DIR}/rgpot/rpc>
           $<BUILD_INTERFACE:${CAPNP_OUTPUT_DIR}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
           $<INSTALL_INTERFACE:include/rgpot/rpc>
           $<INSTALL_INTERFACE:include>)

  # Client Bridge
  if(APPLE)
    add_library(rgpot_client_bridge STATIC CppCore/rgpot/rpc/pot_bridge.cc)
  else()
    add_library(rgpot_client_bridge SHARED CppCore/rgpot/rpc/pot_bridge.cc)
  endif()
  target_link_libraries(rgpot_client_bridge PUBLIC ptlrpc CapnProto::capnp-rpc)
  set_target_properties(rgpot_client_bridge PROPERTIES LINKER_LANGUAGE CXX)

  target_include_directories(
    rgpot_client_bridge
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CppCore>
           $<INSTALL_INTERFACE:include>)
endif()

if(NOT RGPOT_RPC_CLIENT_ONLY)
  # Basic sources
  # The classical pots potserv can serve. Keep this in step with the arms in
  # rgpot/rpc/server.cpp: the server names them unconditionally, so a missing
  # source here surfaces as an undefined constructor at link time rather than
  # as a smaller build.
  set(RGPOT_SOURCES
      CppCore/rgpot/CalculatorGroup.cc
      CppCore/rgpot/PotHelpers.cc
      CppCore/rgpot/units.cc
      CppCore/rgpot/abi/CxxLayout.cc
      CppCore/rgpot/plugin/PluginRegistry.cc
      CppCore/rgpot/plugin/PluginBridge.cc
      CppCore/rgpot/plugin/builtins/lj_plugin.cpp
      CppCore/rgpot/LennardJones/LJPot.cc
      CppCore/rgpot/LennardJones/LJClusterPot.cc
      CppCore/rgpot/Morse/MorsePot.cc
      CppCore/rgpot/ZBL/ZBLPot.cc
      CppCore/rgpot/CuH2/cuh2Utils.cc
      CppCore/rgpot/fortran/FortranPots.cc)

  # NWChemPot / CPMDPot are pure dlopen consumers of the split nwchemc / cpmdc
  # engines (parity with meson). Cap'n Proto params live in the generated schema,
  # so the frontends are only compiled when RGPOT_WITH_RPC enables ptlrpc.
  if(RGPOT_WITH_RPC)
    list(APPEND RGPOT_SOURCES CppCore/rgpot/NWChemPot/NWChemPot.cc
         CppCore/rgpot/CPMDPot/CPMDPot.cc)
  endif()

  if(RGPOT_BUILD_TESTS)
    enable_testing()
  endif()
  find_package(Threads REQUIRED)
  add_library(rgpot_vesin STATIC third_party/vesin/vesin-single-build.cpp)
  add_library(rgpot::vesin ALIAS rgpot_vesin)
  set_target_properties(rgpot_vesin PROPERTIES
    EXPORT_NAME vesin
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden)
  target_include_directories(rgpot_vesin
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/vesin>
           $<INSTALL_INTERFACE:include/rgpot/vesin>)
  target_link_libraries(rgpot_vesin PRIVATE Threads::Threads)
  add_subdirectory(CppCore/rgpot/fortran)
  add_library(rgpot ${RGPOT_SOURCES})
  set_target_properties(rgpot PROPERTIES LINKER_LANGUAGE CXX
                                        POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(rgpot PRIVATE rgpot_fortran_kernels Threads::Threads)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(
      rgpot PRIVATE
      "LINKER:--exclude-libs,$<TARGET_FILE_NAME:rgpot_fortran_kernels>")
    get_target_property(_rgpot_library_type rgpot TYPE)
    if(_rgpot_library_type STREQUAL "STATIC_LIBRARY")
      # Static consumers perform the final link and must hide the kernel
      # archive's internal Fortran entries in their own shared objects.
      target_link_options(
        rgpot INTERFACE
        "$<BUILD_INTERFACE:LINKER:--exclude-libs,$<TARGET_FILE_NAME:rgpot_fortran_kernels>>"
        "$<INSTALL_INTERFACE:LINKER:--exclude-libs,$<TARGET_FILE_NAME:rgpot::rgpot_fortran_kernels>>")
    endif()
  endif()

  # librgpot never links MPI; it loads librgpot_mpi with dlopen when a host
  # asks for calculator groups (CppCore/rgpot/calculator_mpi_abi.h).
  target_link_libraries(rgpot PRIVATE ${CMAKE_DL_LIBS})
  find_package(MPI QUIET COMPONENTS CXX)
  set(RGPOT_HAVE_MPI OFF)
  if(MPI_FOUND)
    # A system MPI wrapper can be found without headers the active
    # compiler can include; only a configuration that compiles counts.
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
    check_cxx_source_compiles(
      "#include <mpi.h>
       int main(int argc, char **argv) { MPI_Init(&argc, &argv); return MPI_Finalize(); }"
      RGPOT_MPI_COMPILES)
    unset(CMAKE_REQUIRED_LIBRARIES)
    set(RGPOT_HAVE_MPI ${RGPOT_MPI_COMPILES})
  endif()
  if(RGPOT_HAVE_MPI)
    add_library(rgpot_mpi SHARED CppCore/rgpot/CalculatorGroupMpi.cc)
    target_include_directories(rgpot_mpi
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/CppCore)
    target_link_libraries(rgpot_mpi PRIVATE MPI::MPI_CXX)
    set_target_properties(rgpot_mpi PROPERTIES CXX_VISIBILITY_PRESET hidden
                                               SOVERSION 3)
  endif()

  if(RGPOT_WITH_CACHE)
    target_sources(rgpot PRIVATE CppCore/rgpot/PotentialCache.cc)
    target_link_libraries(rgpot PUBLIC RocksDB::rocksdb xxHash::xxhash)
    target_compile_definitions(rgpot PUBLIC RGPOT_HAS_CACHE)
  endif()

  if(RGPOT_WITH_RPC)
    target_link_libraries(rgpot PUBLIC ptlrpc)
    target_compile_definitions(rgpot PUBLIC RGPOT_HAS_NWCHEM_FRONTEND
                                            RGPOT_HAS_CPMD_FRONTEND)
    # DynLib / dlopen for libnwchemc.so / libcpmdc.so at runtime
    if(UNIX AND NOT APPLE)
      target_link_libraries(rgpot PUBLIC ${CMAKE_DL_LIBS})
    elseif(UNIX)
      target_link_libraries(rgpot PUBLIC dl)
    endif()
  endif()

  add_library(rgpot::rgpot ALIAS rgpot)

  target_include_directories(
    rgpot PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CppCore>
                 $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                 $<INSTALL_INTERFACE:include>)

  if(UNIX AND NOT APPLE)
    target_link_libraries(rgpot PUBLIC m)
  endif()

  if(NOT RGPOT_PURE_LIB)
    target_link_libraries(rgpot PUBLIC fmt)
    target_compile_definitions(rgpot PUBLIC NOT_PURE_LIB)
  endif()

  if(RGPOT_WITH_XTENSOR)
    target_link_libraries(rgpot PUBLIC xtensor xtensor-blas)
    target_compile_definitions(rgpot PUBLIC WITH_XTENSOR)
  endif()

  if(RGPOT_WITH_EIGEN)
    target_link_libraries(rgpot PUBLIC Eigen3::Eigen)
  endif()

  if(MSVC)
    target_compile_options(rgpot PRIVATE /W4)
  else()
    target_compile_options(rgpot PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()

# The MPI hosts are explicit executables; the serial server and core library
# do not acquire an MPI load dependency.
if(RGPOT_WITH_RPC AND NOT RGPOT_RPC_CLIENT_ONLY)
  add_executable(potserv CppCore/rgpot/rpc/server.cpp)
  target_link_libraries(potserv PRIVATE rgpot::rgpot ptlrpc CapnProto::capnp-rpc)
  if(RGPOT_HAVE_MPI)
    add_executable(potserv_mpi CppCore/rgpot/rpc/server.cpp)
    target_compile_definitions(potserv_mpi PRIVATE RGPOT_POTSERV_MPI)
    target_link_libraries(potserv_mpi PRIVATE rgpot::rgpot ptlrpc
                                            CapnProto::capnp-rpc MPI::MPI_CXX)
    add_executable(nwchem_mpi_force_host CppCore/tools/nwchem_mpi_force_host.cc)
    target_link_libraries(nwchem_mpi_force_host PRIVATE rgpot::rgpot MPI::MPI_CXX)
  endif()
endif()

# --- Installation ---
include(GNUInstallDirs)

if(RGPOT_IS_TOP_LEVEL)
  if(TARGET potserv)
    install(TARGETS potserv RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
  if(TARGET potserv_mpi)
    install(TARGETS potserv_mpi nwchem_mpi_force_host
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
  if(RGPOT_WITH_RPC)
    install(
      TARGETS ptlrpc rgpot_client_bridge
      EXPORT rgpotTargets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    # Install the bridge header in the correct directory structure
    install(FILES CppCore/rgpot/rpc/pot_bridge.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rgpot/rpc)
    install(FILES ${CAPNP_HEADERS} CppCore/rgpot/rpc/Potentials.capnp
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rgpot/rpc)
  endif()

  if(NOT RGPOT_RPC_CLIENT_ONLY)
    install(
      TARGETS rgpot rgpot_fortran_kernels rgpot_vesin
      EXPORT rgpotTargets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      INCLUDES
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES third_party/vesin/vesin.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rgpot/vesin)
    install(TARGETS rgpot_cuh2
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    if(TARGET rgpot_mpi)
      install(TARGETS rgpot_mpi
              LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
              RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
    install(FILES CppCore/rgpot/calculator_mpi_abi.h include/rgpot/plugin.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rgpot)
    install(
      DIRECTORY CppCore/rgpot
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      FILES_MATCHING
      PATTERN "*.hpp"
      PATTERN "*.h")
  endif()

  install(
    EXPORT rgpotTargets
    FILE rgpotTargets.cmake
    NAMESPACE rgpot::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rgpot)

  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rgpotConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/rgpotConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rgpot)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rgpotConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindxxHash.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/rgpotConfig.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/rgpotConfigVersion.cmake"
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rgpot)
endif()

# --- Tests ---
if(RGPOT_BUILD_TESTS)
  enable_testing()
  find_package(Threads REQUIRED)
  add_executable(AotiPublicationTest CppCore/tests/AotiPublicationTest.cc)
  target_include_directories(AotiPublicationTest PRIVATE CppCore)
  target_compile_features(AotiPublicationTest PRIVATE cxx_std_20)
  target_link_libraries(AotiPublicationTest PRIVATE Threads::Threads)
  add_test(NAME AotiPublicationTest COMMAND AotiPublicationTest)
  add_executable(MolecularFrameTest CppCore/tests/MolecularFrameTest.cc)
  target_include_directories(MolecularFrameTest PRIVATE CppCore)
  target_compile_features(MolecularFrameTest PRIVATE cxx_std_20)
  add_test(NAME MolecularFrameTest COMMAND MolecularFrameTest)
  find_package(Catch2 3 REQUIRED)

  if(RGPOT_RPC_CLIENT_ONLY)
    add_executable(BridgeStressTest CppCore/tests/BridgeStressTest.cc)
    target_link_libraries(BridgeStressTest PRIVATE rgpot_client_bridge
                                                   Catch2::Catch2WithMain)
    add_test(NAME BridgeStressTest COMMAND BridgeStressTest)
  endif()

  if(NOT RGPOT_RPC_CLIENT_ONLY)
    function(add_pot_test TEST_NAME SOURCE_FILE)
      add_executable(${TEST_NAME} ${SOURCE_FILE})
      target_link_libraries(${TEST_NAME} PRIVATE rgpot::rgpot
                                                 Catch2::Catch2WithMain)
      set_target_properties(${TEST_NAME} PROPERTIES LINKER_LANGUAGE CXX)
      add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endfunction()

    add_subdirectory(CppCore/tests/plugin)

    add_pot_test(LJClusterPotTest CppCore/tests/LJClusterPotTest.cc)
    add_pot_test(PairListTest CppCore/tests/PairListTest.cc)
    add_pot_test(CxxLayoutTest CppCore/tests/CxxLayoutTest.cc)
    target_link_libraries(CxxLayoutTest PRIVATE ${CMAKE_DL_LIBS})
    add_library(rgpot_layout_fixture SHARED CppCore/tests/layout_fixture.cc)
    target_include_directories(rgpot_layout_fixture
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/CppCore)
    target_link_libraries(rgpot_layout_fixture PRIVATE rgpot::rgpot)
    set_target_properties(rgpot_layout_fixture PROPERTIES PREFIX "lib")
    set_tests_properties(
      CxxLayoutTest
      PROPERTIES ENVIRONMENT
                 "RGPOT_LAYOUT_FIXTURE=$<TARGET_FILE:rgpot_layout_fixture>")
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
      add_test(NAME CxxLayoutVirtualCounts
               COMMAND Python3::Interpreter
                       "${CMAKE_CURRENT_SOURCE_DIR}/CppCore/tests/check_virtual_counts.py")
    endif()
    add_pot_test(MorsePotTest CppCore/tests/MorsePotTest.cc)
    add_pot_test(FortranPotsTest CppCore/tests/FortranPotsTest.cc)
    target_link_libraries(FortranPotsTest PRIVATE Threads::Threads)
    add_pot_test(CuH2PotTest CppCore/tests/CuH2PotTest.cc)
    if(BUILD_SHARED_LIBS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
      add_test(NAME FortranSymbolAudit
               COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/audit_fortran_symbols.sh"
                       $<TARGET_FILE:rgpot>)
    endif()

    if(RGPOT_WITH_CACHE)
      add_pot_test(InvarianceTest CppCore/tests/InvarianceTest.cc)
      add_pot_test(CacheTest CppCore/tests/CacheTest.cc)
    endif()

    if(RGPOT_WITH_RPC)
      add_pot_test(CapnpAdapterTest CppCore/tests/CapnpAdapterTest.cc)
      find_package(Python3 COMPONENTS Interpreter REQUIRED)
      add_library(nwchemc_lifetime_engine SHARED
                  CppCore/tests/nwchemc_lifetime_engine.cc)
      target_include_directories(nwchemc_lifetime_engine PRIVATE CppCore)
      add_executable(NWChemPotLifetimeTest CppCore/tests/NWChemPotLifetimeTest.cc)
      target_link_libraries(NWChemPotLifetimeTest PRIVATE rgpot::rgpot)
      foreach(_lifetime_mode ordinary probe)
        add_test(
          NAME NWChemPotLifetime-${_lifetime_mode}
          COMMAND Python3::Interpreter
                  "${CMAKE_CURRENT_SOURCE_DIR}/CppCore/tests/test_nwchem_lifetime.py"
                  $<TARGET_FILE:NWChemPotLifetimeTest>
                  $<TARGET_FILE:nwchemc_lifetime_engine> ${_lifetime_mode})
      endforeach()
      if(RGPOT_HAVE_MPI)
        add_executable(MpiHostContractTest CppCore/tests/MpiHostContractTest.cc)
        target_include_directories(MpiHostContractTest PRIVATE CppCore)
        target_link_libraries(MpiHostContractTest PRIVATE MPI::MPI_CXX)
        add_library(collective_engine SHARED CppCore/tests/nwchemc_collective_engine.cc)
        target_link_libraries(collective_engine PRIVATE rgpot::rgpot MPI::MPI_CXX)
        add_executable(MpiRpcContractClient CppCore/tests/MpiRpcContractClient.cc)
        target_link_libraries(MpiRpcContractClient PRIVATE rgpot::rgpot CapnProto::capnp-rpc)
        set(_mpi_host_script "${CMAKE_CURRENT_SOURCE_DIR}/CppCore/tests/test_mpi_hosts.py")
        foreach(_mode owned borrowed)
          add_test(NAME MpiHost-${_mode}
                   COMMAND Python3::Interpreter "${_mpi_host_script}" ${_mode}
                           "${MPIEXEC_EXECUTABLE}" $<TARGET_FILE:MpiHostContractTest>)
          set_tests_properties(MpiHost-${_mode} PROPERTIES TIMEOUT 60 PROCESSORS 4)
        endforeach()
        add_test(NAME NWChemMpiForceHost
                 COMMAND Python3::Interpreter "${_mpi_host_script}" force
                         "${MPIEXEC_EXECUTABLE}" $<TARGET_FILE:nwchem_mpi_force_host>
                         $<TARGET_FILE:collective_engine>)
        set_tests_properties(NWChemMpiForceHost PROPERTIES TIMEOUT 60 PROCESSORS 2)
        foreach(_backend NWChem CPMD)
          foreach(_layout serial mpi)
            set(_rpc_server potserv)
            if(_layout STREQUAL "mpi")
              set(_rpc_server potserv_mpi)
            endif()
            add_test(NAME CollectiveRPC-${_backend}-${_layout}
                     COMMAND Python3::Interpreter "${_mpi_host_script}" rpc
                             "${MPIEXEC_EXECUTABLE}" $<TARGET_FILE:${_rpc_server}>
                             $<TARGET_FILE:MpiRpcContractClient>
                             $<TARGET_FILE:collective_engine> ${_backend} ${_layout})
            set_tests_properties(CollectiveRPC-${_backend}-${_layout}
                                 PROPERTIES TIMEOUT 60 PROCESSORS 2)
          endforeach()
        endforeach()
      endif()
    endif()
  endif()
endif()
