cmake_minimum_required(VERSION 3.16)

project(
  potlib
  VERSION 0.1.0
  DESCRIPTION "Cpp core for potential energy surface functionality"
  HOMEPAGE_URL "https://github.com/theochemui/potlib"
  LANGUAGES C CXX)

# --- Options ---
option(POTLIB_RPC_CLIENT_ONLY "Build ONLY the RPC client bridge (lightweight)"
       OFF)

if(POTLIB_RPC_CLIENT_ONLY)
  set(CMAKE_CXX_STANDARD 17) # C++17 is sufficient for the RPC bridge
  option(POTLIB_WITH_RPC "Build with Cap'n Proto RPC support" ON)

  # Force disable heavy dependencies
  set(POTLIB_WITH_XTENSOR OFF)
  set(POTLIB_WITH_EIGEN OFF)
  set(POTLIB_WITH_CACHE OFF)
  set(POTLIB_PURE_LIB ON)
  set(POTLIB_BUILD_EXAMPLES OFF)
else()
  set(CMAKE_CXX_STANDARD 20)
  option(POTLIB_WITH_XTENSOR "Build with xtensor support" OFF)
  option(POTLIB_WITH_EIGEN "Build with Eigen3 support" OFF)
  option(POTLIB_PURE_LIB "Build as a pure library (no fmt dependency)" ON)
  option(POTLIB_WITH_CACHE "Build with RocksDB caching support" ON)
  option(POTLIB_WITH_RPC "Build with Cap'n Proto RPC support" OFF)
  option(POTLIB_BUILD_EXAMPLES "Build examples" OFF)
endif()

option(POTLIB_BUILD_TESTS "Build tests" OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
include(FetchContent)

# 1. RPC Dependency
if(POTLIB_WITH_RPC)
  find_package(CapnProto REQUIRED)
endif()

# 1. Server/Physics Dependencies
if(NOT POTLIB_RPC_CLIENT_ONLY)
  # Check for Fortran before assuming it exists
  include(CheckLanguage)
  check_language(Fortran)
  if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    set(POTLIB_HAS_FORTRAN TRUE)
    message(STATUS "Fortran found. CuH2 potential enabled.")
  else()
    set(POTLIB_HAS_FORTRAN FALSE)
    message(STATUS "Fortran NOT found. CuH2 potential disabled.")
  endif()

  if(POTLIB_HAS_FORTRAN)
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

  if(POTLIB_WITH_CACHE)
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

  if(NOT POTLIB_PURE_LIB)
    find_package(fmt REQUIRED)
  endif()

  if(POTLIB_WITH_XTENSOR)
    find_package(xtensor REQUIRED)
    find_package(xtensor-blas REQUIRED)
  endif()

  if(POTLIB_WITH_EIGEN)
    find_package(Eigen3 3.4.0 REQUIRED)
  endif()
endif()

# --- Build Targets ---

if(POTLIB_WITH_RPC)
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
  add_library(pot_client_bridge SHARED CppCore/rgpot/rpc/pot_bridge.cc)
  target_link_libraries(pot_client_bridge PUBLIC ptlrpc CapnProto::capnp-rpc)

  target_include_directories(
    pot_client_bridge
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CppCore>
           $<INSTALL_INTERFACE:include>)
endif()

if(NOT POTLIB_RPC_CLIENT_ONLY)
  # Basic sources
  set(POT_SOURCES CppCore/rgpot/PotHelpers.cc
                  CppCore/rgpot/LennardJones/LJPot.cc)

  if(POTLIB_HAS_FORTRAN)
    list(APPEND POT_SOURCES CppCore/rgpot/CuH2/CuH2Pot.cc
         CppCore/rgpot/CuH2/cuh2Utils.cc)
  endif()

  add_library(pot ${POT_SOURCES})

  if(POTLIB_WITH_CACHE)
    target_sources(pot PRIVATE CppCore/rgpot/PotentialCache.cc)
    target_link_libraries(pot PUBLIC RocksDB::rocksdb xxHash::xxhash)
    target_compile_definitions(pot PUBLIC POT_HAS_CACHE)
  endif()

  if(POTLIB_WITH_RPC)
    target_link_libraries(pot PUBLIC ptlrpc)
  endif()

  add_library(pot::pot ALIAS pot)

  target_include_directories(
    pot PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CppCore>
               $<INSTALL_INTERFACE:include>)

  if(POTLIB_HAS_FORTRAN)
    target_link_libraries(pot PUBLIC fortcuh2)
    target_compile_definitions(pot PUBLIC POT_HAS_CUH2)
  endif()

  if(NOT POTLIB_PURE_LIB)
    target_link_libraries(pot PUBLIC fmt)
    target_compile_definitions(pot PUBLIC NOT_PURE_LIB)
  endif()

  if(POTLIB_WITH_XTENSOR)
    target_link_libraries(pot PUBLIC xtensor xtensor-blas)
    target_compile_definitions(pot PUBLIC WITH_XTENSOR)
  endif()

  if(POTLIB_WITH_EIGEN)
    target_link_libraries(pot PUBLIC Eigen3::Eigen)
  endif()

  if(MSVC)
    target_compile_options(pot PRIVATE /W4)
  else()
    target_compile_options(pot PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()

# --- Installation ---
include(GNUInstallDirs)

if(POTLIB_WITH_RPC)
  install(
    TARGETS ptlrpc pot_client_bridge
    EXPORT potTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  # Install the bridge header in the correct directory structure
  install(FILES CppCore/rgpot/rpc/pot_bridge.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rgpot/rpc)
endif()

if(NOT POTLIB_RPC_CLIENT_ONLY)
  install(
    TARGETS pot
    EXPORT potTargets
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
  EXPORT potTargets
  FILE potConfig.cmake
  NAMESPACE pot::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pot)

# --- Tests ---
if(POTLIB_BUILD_TESTS)
  enable_testing()
  find_package(Catch2 3 REQUIRED)

  if(POTLIB_WITH_RPC)
    add_executable(BridgeStressTest CppCore/tests/BridgeStressTest.cc)
    target_link_libraries(BridgeStressTest PRIVATE pot_client_bridge
                                                   Catch2::Catch2WithMain)
    add_test(NAME BridgeStressTest COMMAND BridgeStressTest)
  endif()

  if(NOT POTLIB_RPC_CLIENT_ONLY)
    function(add_pot_test TEST_NAME SOURCE_FILE)
      add_executable(${TEST_NAME} ${SOURCE_FILE})
      target_link_libraries(${TEST_NAME} PRIVATE pot::pot
                                                 Catch2::Catch2WithMain)
      add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endfunction()

    if(POTLIB_HAS_FORTRAN)
      add_pot_test(CuH2Test CppCore/tests/CuH2PotTest.cc)
    endif()

    if(POTLIB_WITH_CACHE)
      add_pot_test(InvarianceTest CppCore/tests/InvarianceTest.cc)
      add_pot_test(CacheTest CppCore/tests/CacheTest.cc)
    endif()

    if(POTLIB_WITH_RPC)
      add_pot_test(CapnpAdapterTest CppCore/tests/CapnpAdapterTest.cc)
      add_executable(potserv CppCore/rgpot/rpc/server.cpp)
      target_link_libraries(potserv PRIVATE pot::pot ptlrpc
                                            CapnProto::capnp-rpc)
    endif()
  endif()
endif()
