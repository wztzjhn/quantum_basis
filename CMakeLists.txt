cmake_minimum_required(VERSION 3.12)

project(qbasis VERSION 0.1.0 DESCRIPTION "Exact diagonalization library")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# -- COMPILE OPTIONS -------------------------------------------
# Performance/debug options
set(GCC_COVERAGE_COMPILE_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g ${GCC_COVERAGE_COMPILE_FLAGS}")
add_definitions(-D_GLIBCXX_DEBUG=1) # libstdc++
add_definitions(-D_LIBCPP_DEBUG2=1) # libc++


# -- C++17 -------------------------------------------
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(NOT COMPILER_SUPPORTS_CXX17)
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Aborting.")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()


# -- DEPENDENCIES -------------------------------------------
#ARPACK
find_package(ARPACK REQUIRED)
include_directories(SYSTEM ${ARPACK_INCLUDE_DIRS})
set(LIBS ${LIBS} ${ARPACK_LIBRARIES})

# Boost
find_package(Boost REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

# MKL
set(MKL_ARCH "intel64")
set(MKL_LINK "dynamic")
set(MKL_THREADING "sequential")
set(MKL_INTERFACE_FULL "intel_ilp64")
add_compile_definitions(MKL_ILP64)
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
