cmake_minimum_required(VERSION 3.12)

project(qbasis)
set(QBASIS_VERSION 0.1.0)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

SET(CMAKE_INSTALL_PREFIX "$ENV{HOME}/installs")
#SET(CMAKE_LIBRARY_PATH "$ENV{HOME}/installs/lib")

# -- COMPILE OPTIONS -------------------------------------------
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
set(INSTALL_CMAKE_DIR lib/CMake/qbasis CACHE PATH "Installation directory for CMake files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()


# Performance/debug options
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND "${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -gdwarf-3 -DDEBUG")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -gdwarf-3 -debug all -traceback -fstack-security-check")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -Wall -Wextra")
endif()
add_definitions(-D_GLIBCXX_DEBUG=1) # libstdc++
add_definitions(-D_LIBCPP_DEBUG2=1) # libc++

# -- C++17 -------------------------------------------

# C++17
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(NOT COMPILER_SUPPORTS_CXX17)
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Aborting.")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# Mac fixes
if (APPLE)
  cmake_policy(SET CMP0042 NEW) # @rpath
  #set(LIBS ${LIBS} "-framework Accelerate")
  #set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
endif()

# -- DEPENDENCIES -------------------------------------------

#ARPACK
find_package(ARPACK REQUIRED)
if (ARPACK_FOUND)
  include_directories(SYSTEM ${ARPACK_INCLUDE_DIRS})
  include_directories(SYSTEM ${ARPACK_INCLUDE_DIRS}/../examples/areig)
  set(LIBS ${LIBS} ${ARPACK_LIBRARIES})
  add_definitions(-DWITH_ARPACK)
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lgfortran")
  endif()
endif()

# Boost
find_package(Boost REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

# MKL
set(MKL_ARCH "intel64")
set(MKL_LINK "dynamic")
set(MKL_THREADING "sequential")
set(MKL_INTERFACE_FULL "intel_lp64")
find_package(MKL CONFIG REQUIRED)
message("MKL_INCLUDE: ${MKL_INCLUDE}")

# OPENMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# -- BUILD -------------------------------------------
include_directories(
  "${PROJECT_SOURCE_DIR}"   # to find src/qbasis.h
  "${PROJECT_BINARY_DIR}")
add_subdirectory(src)

# -- INSTALLATION -------------------------------------------

# Generate qbasisTargets.cmake suitable for use by downstreams
export(TARGETS qbasis FILE "${PROJECT_BINARY_DIR}/qbasisTargets.cmake")

# Register the build-tree with a global CMake-registry
export(PACKAGE qbasis)


# Create the FooBarConfig.cmake and FooBarConfigVersion files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")
# ... for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/src")
configure_file(qbasisConfig.cmake.in "${PROJECT_BINARY_DIR}/qbasisConfig.cmake" @ONLY)
# ... for the install tree
set(CONF_INCLUDE_DIRS "${INSTALL_INCLUDE_DIR}")
configure_file(qbasisConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/qbasisConfig.cmake" @ONLY)
# ... for both
configure_file(qbasisConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/qbasisConfigVersion.cmake" @ONLY)

# Install qbasisConfig.cmake and qbasisConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/qbasisConfig.cmake"
  "${PROJECT_BINARY_DIR}/qbasisConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

# Install qbasisTargets.cmake
install(EXPORT qbasisTargets DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)


# -- REPORT -------------------------------------------

message("")
message("Configuration results")
message("---------------------")
message("Operating System          : ${CMAKE_SYSTEM_NAME}")
message("C++ compiler              : ${CMAKE_CXX_COMPILER}")
message("C++ compiler id           : ${CMAKE_CXX_COMPILER_ID}")
message("C++ compiler flags        : ${CMAKE_CXX_FLAGS}")

get_property(INC_DIRS DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("LIBS:     ${LIBS}")
message("INCLUDES: ${INC_DIRS}")
message("")
