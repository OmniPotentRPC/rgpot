cmake_minimum_required(VERSION 3.16)

project(
  rgpot
  VERSION 0.1.0
  DESCRIPTION "Cpp core for potential energy surface functionality"
  HOMEPAGE_URL "https://github.com/OmniPotentRPC/rgpot"
  LANGUAGES C CXX)

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
  option(RGPOT_BUILD_EXAMPLES "Build examples" OFF)
endif()

option(RGPOT_BUILD_TESTS "Build tests" OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
include(FetchContent)

# 1. RPC Dependency
if(RGPOT_WITH_RPC)
  find_package(CapnProto REQUIRED)
endif()

# 1. Server/Physics Dependencies
if(NOT RGPOT_RPC_CLIENT_ONLY)
  # Check for Fortran before assuming it exists
  include(CheckLanguage)
  check_language(Fortran)
  if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    set(RGPOT_HAS_FORTRAN TRUE)
    message(STATUS "Fortran found. CuH2 potential enabled.")
  else()
    set(RGPOT_HAS_FORTRAN FALSE)
    message(STATUS "Fortran NOT found. CuH2 potential disabled.")
  endif()

  if(RGPOT_HAS_FORTRAN)
    find_package(fortcuh2 QUIET)
    if(NOT fortcuh2_FOUND)
      message(STATUS "fortcuh2 not found, attempting to fetch...")
      FetchContent_Declare(
        fortcuh2
        GIT_REPOSITORY https://github.com/theochemui/fortran_cuh2_src.git
        GIT_TAG main)
      FetchContent_MakeAvailable(fortcuh2)
    endif()
  endif()

  if(RGPOT_WITH_CACHE)
    find_package(RocksDB REQUIRED)
    find_package(xxHash CONFIG QUIET)
    if(NOT xxHash_FOUND)
      message(
        STATUS "xxHash not found via find_package, fetching from source...")
      FetchContent_Declare(
        xxHash
        GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
        GIT_TAG v0.8.3)
      FetchContent_MakeAvailable(xxHash)
      if(TARGET xxhash AND NOT TARGET xxHash::xxhash)
        add_library(xxHash::xxhash ALIAS xxhash)
      endif()
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

  add_library(ptlrpc SHARED ${CAPNP_SOURCES})
  target_link_libraries(ptlrpc PUBLIC CapnProto::capnp-rpc)

  # Ensure the generated headers are in the include path
  target_include_directories(
    ptlrpc
    PUBLIC $<BUILD_INTERFACE:${CAPNP_OUTPUT_DIR}/rgpot/rpc>
           $<BUILD_INTERFACE:${CAPNP_OUTPUT_DIR}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

  # Client Bridge
  add_library(rgpot_client_bridge SHARED CppCore/rgpot/rpc/pot_bridge.cc)
  target_link_libraries(rgpot_client_bridge PUBLIC ptlrpc CapnProto::capnp-rpc)

  target_include_directories(
    rgpot_client_bridge
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CppCore>
           $<INSTALL_INTERFACE:include>)
endif()

if(NOT RGPOT_RPC_CLIENT_ONLY)
  # Basic sources
  set(RGPOT_SOURCES CppCore/rgpot/PotHelpers.cc
                    CppCore/rgpot/LennardJones/LJPot.cc)

  if(RGPOT_HAS_FORTRAN)
    list(APPEND RGPOT_SOURCES CppCore/rgpot/CuH2/CuH2Pot.cc
         CppCore/rgpot/CuH2/cuh2Utils.cc)
  endif()

  add_library(rgpot ${RGPOT_SOURCES})

  if(RGPOT_WITH_CACHE)
    target_sources(rgpot PRIVATE CppCore/rgpot/PotentialCache.cc)
    target_link_libraries(rgpot PUBLIC RocksDB::rocksdb xxhash)
    target_compile_definitions(rgpot PUBLIC RGPOT_HAS_CACHE)
  endif()

  if(RGPOT_WITH_RPC)
    target_link_libraries(rgpot PUBLIC ptlrpc)
  endif()

  add_library(rgpot::rgpot ALIAS rgpot)

  target_include_directories(
    pot PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CppCore>
               $<INSTALL_INTERFACE:include>)

  if(RGPOT_HAS_FORTRAN)
    target_link_libraries(rgpot PUBLIC fortcuh2)
    target_compile_definitions(rgpot PUBLIC POT_HAS_FORTRAN)
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

# --- Installation ---
include(GNUInstallDirs)

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
endif()

if(NOT RGPOT_RPC_CLIENT_ONLY)
  install(
    TARGETS rgpot
    EXPORT rgpotTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(
    DIRECTORY CppCore/rgpot
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "*.hpp")
endif()

install(
  EXPORT rgpotTargets
  FILE rgpotConfig.cmake
  NAMESPACE rgpot::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pot)

# --- Tests ---
if(RGPOT_BUILD_TESTS)
  enable_testing()
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
      add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endfunction()

    if(RGPOT_HAS_FORTRAN)
      add_pot_test(CuH2Test CppCore/tests/CuH2PotTest.cc)
    endif()

    if(RGPOT_WITH_CACHE)
      add_pot_test(InvarianceTest CppCore/tests/InvarianceTest.cc)
      add_pot_test(CacheTest CppCore/tests/CacheTest.cc)
    endif()

    if(RGPOT_WITH_RPC)
      add_pot_test(CapnpAdapterTest CppCore/tests/CapnpAdapterTest.cc)
      add_executable(potserv CppCore/rgpot/rpc/server.cpp)
      target_link_libraries(potserv PRIVATE rgpot::rgpot ptlrpc
                                            CapnProto::capnp-rpc)
    endif()
  endif()
endif()
