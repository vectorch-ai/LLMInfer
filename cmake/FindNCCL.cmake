# Find the nccl libraries
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT: Base directory where all NCCL components are found
#  NCCL_INCLUDE_DIR: Directory where NCCL header is found
#  NCCL_LIB_DIR: Directory where NCCL library is found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES
#
# Adapted from https://github.com/pytorch/pytorch/blob/master/cmake/Modules/FindNCCL.cmake

set(NCCL_INCLUDE_DIR $ENV{NCCL_INCLUDE_DIR} CACHE PATH "Folder contains NVIDIA NCCL headers")
set(NCCL_LIB_DIR $ENV{NCCL_LIB_DIR} CACHE PATH "Folder contains NVIDIA NCCL libraries")
set(NCCL_VERSION $ENV{NCCL_VERSION} CACHE STRING "Version of NCCL to build with")

list(APPEND NCCL_ROOT $ENV{NCCL_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})
# Compatible layer for CMake <3.12. NCCL_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${NCCL_ROOT})

find_path(NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS
    ${NCCL_INCLUDE_DIR}
    $ENV{CUDA_HOME}
    $ENV{CUDA_PATH}
    $ENV{CUDA_TOOLKIT_ROOT_DIR}
    $ENV{NCCL}
    /usr/local/cuda
    /usr
  PATH_SUFFIXES
    include
)

if (USE_STATIC_NCCL)
  MESSAGE(STATUS "USE_STATIC_NCCL is set. Linking with static NCCL library.")
  SET(NCCL_LIBNAME "nccl_static")
  if (NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
else()
  SET(NCCL_LIBNAME "nccl")
  if (NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
endif()

# Read version from header
if(EXISTS "${NCCL_INCLUDE_DIRS}/nccl.h")
  file(READ ${NCCL_INCLUDE_DIRS}/nccl.h NCCL_HEADER_CONTENTS)
endif()
if(NCCL_HEADER_CONTENTS)
  string(REGEX MATCH "define NCCL_MAJOR * +([0-9]+)"
               NCCL_VERSION_MAJOR "${NCCL_HEADER_CONTENTS}")
  string(REGEX REPLACE "define NCCL_MAJOR * +([0-9]+)" "\\1"
               NCCL_VERSION_MAJOR "${NCCL_VERSION_MAJOR}")
  string(REGEX MATCH "define NCCL_MINOR * +([0-9]+)"
               NCCL_VERSION_MINOR "${NCCL_HEADER_CONTENTS}")
  string(REGEX REPLACE "define NCCL_MINOR * +([0-9]+)" "\\1"
    NCCL_VERSION_MINOR "${NCCL_VERSION_MINOR}")
  string(REGEX MATCH "define NCCL_PATCH * +([0-9]+)"
    NCCL_VERSION_PATCH "${NCCL_HEADER_CONTENTS}")
  string(REGEX REPLACE "define NCCL_PATCH * +([0-9]+)" "\\1"
    NCCL_VERSION_PATCH "${NCCL_VERSION_PATCH}")
  if(NOT NCCL_VERSION_MAJOR)
    set(NCCL_VERSION "?")
  else()
    set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
  endif()
endif()

find_library(NCCL_LIBRARIES
  NAMES ${NCCL_LIBNAME}
  HINTS
    ${NCCL_LIB_DIR}
    ${NCCL_INCLUDE_DIR}
    $ENV{CUDA_HOME}
    $ENV{CUDA_PATH}
    $ENV{CUDA_TOOLKIT_ROOT_DIR}
    /usr/local/cuda
    /usr/lib/x86_64-linux-gnu/
  PATH_SUFFIXES
    lib
    lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
  REQUIRED_VARS NCCL_INCLUDE_DIRS NCCL_LIBRARIES
  VERSION_VAR   NCCL_VERSION)

if(NOT NCCL_FOUND)
  if(NCCL_FIND_REQUIRED)
    message(FATAL_ERROR "Cannot find NCCL!")
  else()
    message(WARNING "NCCL is not found!")
  endif()
else()
  message(STATUS "Found NCCL ${NCCL_VERSION} (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
  if(NOT TARGET NCCL::nccl)
    add_library(NCCL::nccl UNKNOWN IMPORTED)
    set_target_properties(NCCL::nccl PROPERTIES
      IMPORTED_LOCATION "${NCCL_LIBRARIES}"
      INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIRS}"
    )
  endif()
endif()
