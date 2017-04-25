cmake_minimum_required(VERSION 2.8)

project(qbasis)
set(QBASIS_VERSION 0.1.0)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

SET(CMAKE_INSTALL_PREFIX "$ENV{HOME}/installs")
SET(CMAKE_LIBRARY_PATH "$ENV{HOME}/installs/lib")

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


# C++11
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(NOT COMPILER_SUPPORTS_CXX11)
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Aborting.")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
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

# Boost
#find_package(Boost 1.56.0 REQUIRED)
#include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

#MKL
find_package(MKL REQUIRED)
include_directories(SYSTEM ${MKL_INCLUDE_DIRS})
#temporarily use this ugly way
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
#mac clang
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND "${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
  #set(LIBS "-L${MKL_ROOT}/lib -Wl,-rpath,${MKL_ROOT}/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl")
  set(LIBS "-L/opt/intel/lib -L${MKL_ROOT}/lib -Wl,-rpath,${MKL_ROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl")
#mac gcc
elseif(CMAKE_COMPILER_IS_GNUCXX AND "${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
  set(LIBS "-L${MKL_ROOT}/lib -Wl,-rpath,${MKL_ROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl")
#linux gcc
elseif(CMAKE_COMPILER_IS_GNUCXX AND "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
  #set(LIBS "-L${MKL_ROOT}/lib/intel64 -Wl,-rpath,${MKL_ROOT}/lib -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl")
  #set(LIBS "-L${MKL_ROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")
  set(LIBS "-L${MKL_ROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl")
#linux intel
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" AND "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel")
  set(LIBS "-liomp5 -lpthread -lm -ldl")
#  set(LIBS "-L${MKL_ROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl")
#elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
#  set(LIBS "-L${MKL_ROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl")
endif()

#ARPACK
find_package(ARPACK REQUIRED)
if (ARPACK_FOUND)
  include_directories(SYSTEM ${ARPACK_INCLUDE_DIRS})
  include_directories(SYSTEM ${ARPACK_INCLUDE_DIRS}/../examples/areig)
  set(LIBS "${LIBS} ${ARPACK_LIBRARIES}")
  add_definitions(-DWITH_ARPACK)
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lgfortran")
  endif()
endif()

# Add openmp if Linux platform
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  set(LIBS "${LIBS} -fopenmp")
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
