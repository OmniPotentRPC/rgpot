cmake_minimum_required(VERSION 3.16)

project(potlib
    VERSION 0.1.0
    DESCRIPTION "Cpp core for potential energy surface functionality"
    HOMEPAGE_URL "https://github.com/theochemui/potlib"
    LANGUAGES C CXX Fortran
)

# --- Options ---
option(POTLIB_WITH_XTENSOR "Build with xtensor support" OFF)
option(POTLIB_WITH_EIGEN "Build with Eigen3 support" OFF)
option(POTLIB_PURE_LIB "Build as a pure library (no fmt dependency)" ON)
option(POTLIB_WITH_RPC "Build with Cap'n Proto RPC support" OFF)
option(POTLIB_BUILD_TESTS "Build tests" OFF)
option(POTLIB_BUILD_EXAMPLES "Build examples" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
include(FetchContent)
find_package(RocksDB REQUIRED)

find_package(xxHash CONFIG QUIET)
if(NOT xxHash_FOUND)
    message(STATUS "xxHash not found via find_package, fetching from source...")
    FetchContent_Declare(
        xxHash
        GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
        GIT_TAG v0.8.3
    )
    FetchContent_MakeAvailable(xxHash)

    # Ensure the namespaced target exists for consistency
    if(TARGET xxhash AND NOT TARGET xxHash::xxhash)
        add_library(xxHash::xxhash ALIAS xxhash)
        message(STATUS "Created alias xxHash::xxhash -> xxhash")
    endif()
endif()

# 3. Fortcuh2 (Required - User defined)
# We assume this is available via find_package (if installed)
# or FetchContent (if subproject).
find_package(fortcuh2 QUIET)
if(NOT fortcuh2_FOUND)
    message(STATUS "fortcuh2 not found, attempting to fetch...")
    # Update the GIT_REPOSITORY to the actual location of your fortran project
    FetchContent_Declare(
        fortcuh2
        GIT_REPOSITORY https://github.com/theochemui/fortran_cuh2_src.git
        GIT_TAG main
    )
    FetchContent_MakeAvailable(fortcuh2)
endif()

if(NOT POTLIB_PURE_LIB OR POTLIB_BUILD_EXAMPLES)
    find_package(fmt REQUIRED)
endif()

if(POTLIB_WITH_XTENSOR)
    find_package(xtensor REQUIRED)
    find_package(xtensor-blas REQUIRED)
endif()

if(POTLIB_WITH_EIGEN)
    find_package(Eigen3 3.4.0 REQUIRED)
endif()

if(POTLIB_WITH_RPC)
    find_package(CapnProto REQUIRED)
endif()

# --- Library Definition ---
add_library(pot
    CppCore/rgpot/PotHelpers.cc
    CppCore/rgpot/PotentialCache.cc
    CppCore/rgpot/LennardJones/LJPot.cc
    CppCore/rgpot/CuH2/CuH2Pot.cc
    CppCore/rgpot/CuH2/cuh2Utils.cc
)

if(POTLIB_WITH_RPC)
    set(CAPNP_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/CppCore")
    file(MAKE_DIRECTORY "${CAPNP_OUTPUT_DIR}/rgpot/rpc")
    
    capnp_generate_cpp(CAPNP_SOURCES CAPNP_HEADERS
      CppCore/rgpot/rpc/Potentials.capnp)
    add_library(ptlrpc SHARED ${CAPNP_SOURCES})
    target_link_libraries(ptlrpc PUBLIC CapnProto::capnp-rpc)
    target_include_directories(ptlrpc PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/CppCore>)
    target_link_libraries(pot PUBLIC ptlrpc)
endif()

add_library(pot::pot ALIAS pot)

# --- Include Directories ---
target_include_directories(pot PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CppCore>
    $<INSTALL_INTERFACE:include>
)

# --- Linking ---
target_link_libraries(pot PUBLIC
    RocksDB::rocksdb
    xxhash
    fortcuh2
)

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

if(APPLE)
    target_compile_definitions(pot PUBLIC OSX)
    # Meson adds -faligned-allocation for clang on mac
    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        target_compile_options(pot PRIVATE -faligned-allocation)
    endif()
endif()

# --- Compiler Warnings (Matching Meson) ---
if(MSVC)
    target_compile_options(pot PRIVATE /W4)
else()
    target_compile_options(pot PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- Installation & Export ---
include(GNUInstallDirs)

install(TARGETS pot EXPORT potTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY CppCore/rgpot DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT potTargets
    FILE potConfig.cmake
    NAMESPACE pot::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pot
)

# --- Tests ---
if(POTLIB_BUILD_TESTS)
    enable_testing()
    find_package(Catch2 3 REQUIRED)

    function(add_pot_test TEST_NAME SOURCE_FILE)
        add_executable(${TEST_NAME} ${SOURCE_FILE})
        target_link_libraries(${TEST_NAME} PRIVATE pot::pot Catch2::Catch2WithMain)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endfunction()

    add_pot_test(CuH2Test CppCore/tests/CuH2PotTest.cc)
    add_pot_test(CacheTest CppCore/tests/CacheTest.cc)
    add_pot_test(InvarianceTest CppCore/tests/InvarianceTest.cc)
    if(POTLIB_WITH_RPC)
        add_pot_test(CapnpAdapterTest CppCore/tests/CapnpAdapterTest.cc)
    endif()
endif()

# --- Examples ---
if(POTLIB_BUILD_EXAMPLES)
    # 1. Standard Call
    add_executable(call_cuh2 CppCore/examples/call_cuh2.cc)
    target_link_libraries(call_cuh2 PRIVATE pot::pot fmt)

    if(POTLIB_WITH_RPC)
        add_executable(potserv CppCore/rgpot/rpc/server.cpp)
        target_link_libraries(potserv PRIVATE pot::pot ptlrpc CapnProto::capnp-rpc)
    endif()

    # 2. Eigen Examples
    if(POTLIB_WITH_EIGEN)
        add_executable(eigen_call_ljpot CppCore/examples/eigen_call_ljpot.cc)
        target_link_libraries(eigen_call_ljpot PRIVATE pot::pot fmt Eigen3::Eigen)

        add_executable(eigen_call_cuh2 CppCore/examples/eigen_call_cuh2.cc)
        target_link_libraries(eigen_call_cuh2 PRIVATE pot::pot fmt Eigen3::Eigen)
    endif()

    # 3. Xtensor Examples
    if(POTLIB_WITH_XTENSOR)
        add_executable(xt_call_cuh2 CppCore/examples/xt_call_cuh2.cc)
        target_link_libraries(xt_call_cuh2 PRIVATE pot::pot fmt xtensor xtensor-blas)
    endif()
endif()
