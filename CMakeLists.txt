cmake_minimum_required(VERSION 2.8)

project(qbasis)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# -- COMPILE OPTIONS -------------------------------------------

# Performance/debug options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
add_definitions(-D_GLIBCXX_DEBUG=1) # libstdc++
add_definitions(-D_LIBCPP_DEBUG2=1) # libc++

enable_language(Fortran)
get_filename_component (FORTRAN_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wall -O0 -fcheck=all -mcmodel=large")


# -- C++11 -------------------------------------------

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


# -- DEPENDENCIES -------------------------------------------

# Boost
find_package(Boost 1.56.0 REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

#MKL
find_package(MKL)
include_directories(SYSTEM ${MKL_INCLUDE_DIRS})
#temporarily use this ugly way
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
set(LIBS "-L${MKL_ROOT}/lib -Wl,-rpath,${MKL_ROOT}/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl")

# ARPACK (long term goal: make this option optional)
include_directories(SYSTEM $HOME/opt/ARPACK)
set(LIBS "${LIBS} -L$HOME/opt/ARPACK -larpack -L/usr/local/Cellar/gcc/6.2.0/lib/gcc/6 -lgfortran")

# -- BUILD -------------------------------------------

include_directories(src)
add_subdirectory(src)


# -- REPORT -------------------------------------------

message("")
message("Configuration results")
message("---------------------")
message("C++ compiler            : ${CMAKE_CXX_COMPILER}")
message("C++ compiler flags      : ${CMAKE_CXX_FLAGS}")
message("Fortran compiler        : ${CMAKE_Fortran_COMPILER}")
message("Fortran compiler flags  : ${CMAKE_Fortran_FLAGS}")

get_property(INC_DIRS DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("LIBS:     ${LIBS}")
message("INCLUDES: ${INC_DIRS}")
message("")
