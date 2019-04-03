# - Find Intel MKL
# Find the MKL libraries
#   MKL_FOUND            : True if MKL_INCLUDE_DIR are found
#   MKL_INCLUDE_DIR      : where to find mkl.h, etc.
#   MKL_INCLUDE_DIRS     : set when MKL_INCLUDE_DIR found
#   MKL_LIBRARIES        : the library to link against.


set(MKL_ROOT $ENV{MKLROOT})

# Find include dir
IF(MKL_INCLUDE_DIRS)
  find_path(MKL_INCLUDE_DIR mkl.h ${MKL_INCLUDE_DIRS})
  FIND_LIBRARY(MKL_LIBRARY mkl ${MKL_LIBRARY_DIRS})
ELSE(MKL_INCLUDE_DIRS)
    SET(TRIAL_PATHS
      /usr/include
      /usr/local/include
      ${MKL_ROOT}/include
    )
    FIND_PATH(MKL_INCLUDE_DIR mkl.h ${TRIAL_PATHS})
ENDIF(MKL_INCLUDE_DIRS)
set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR} )

SET(MKL_FOUND FALSE)
IF(MKL_INCLUDE_DIR)
  MESSAGE(STATUS "MKL_INCLUDE_DIR=${MKL_INCLUDE_DIR}")
  SET(MKL_FOUND TRUE)
ENDIF()

MARK_AS_ADVANCED(
   MKL_INCLUDE_DIR
   MKL_FOUND
)
