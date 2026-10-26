# Prefer xxHash's exported target; system packages may supply only the
# library and header, so retain the same target name for that layout.
find_package(xxHash CONFIG QUIET)
if(TARGET xxHash::xxhash)
  set(xxHash_FOUND TRUE)
  return()
endif()

find_path(xxHash_INCLUDE_DIR NAMES xxhash.h)
find_library(xxHash_LIBRARY NAMES xxhash)
if(xxHash_INCLUDE_DIR)
  file(STRINGS "${xxHash_INCLUDE_DIR}/xxhash.h" _xxhash_version_lines
    REGEX "^#define XXH_VERSION_(MAJOR|MINOR|RELEASE) +[0-9]+")
  foreach(component MAJOR MINOR RELEASE)
    foreach(line IN LISTS _xxhash_version_lines)
      if(line MATCHES "^#define XXH_VERSION_${component} +([0-9]+)")
        set(_xxhash_${component} "${CMAKE_MATCH_1}")
      endif()
    endforeach()
  endforeach()
  set(xxHash_VERSION "${_xxhash_MAJOR}.${_xxhash_MINOR}.${_xxhash_RELEASE}")
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(xxHash
  REQUIRED_VARS xxHash_LIBRARY xxHash_INCLUDE_DIR
  VERSION_VAR xxHash_VERSION)
mark_as_advanced(xxHash_LIBRARY xxHash_INCLUDE_DIR)
if(xxHash_FOUND AND NOT TARGET xxHash::xxhash)
  add_library(xxHash::xxhash UNKNOWN IMPORTED)
  set_target_properties(xxHash::xxhash PROPERTIES
    IMPORTED_LOCATION "${xxHash_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${xxHash_INCLUDE_DIR}")
endif()
