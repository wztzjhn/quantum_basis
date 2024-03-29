# ARPACK_INCLUDE_DIR = arpack-ng/arpack.hpp
# ARPACK_LIBRARIES = libarpackILP64.so
# ARPACK_FOUND = true if arpack is found

IF(ARPACK_INCLUDE_DIRS)
  FIND_PATH(ARPACK_INCLUDE_DIR arpack-ng/arpack.hpp ${ARPACK_INCLUDE_DIRS})
  FIND_LIBRARY(ARPACK_LIBRARY arpackILP64 ${ARPACK_LIBRARY_DIRS})
ELSE(ARPACK_INCLUDE_DIRS)
    SET(TRIAL_PATHS
      $ENV{HOME}/installs/include
      /usr/include
      /usr/local/include
    )
    SET(TRIAL_LIBRARY_PATHS
      $ENV{HOME}/installs/lib
      $ENV{HOME}/installs/lib64
      /usr/lib
      /usr/lib64
      /usr/lib/x86_64_linux-gnu
      /usr/local/lib
      /usr/local/lib64
    )
    
    FIND_PATH(ARPACK_INCLUDE_DIR arpack-ng/arpack.hpp ${TRIAL_PATHS})
    FIND_LIBRARY(ARPACK_LIBRARY arpackILP64 ${TRIAL_LIBRARY_PATHS})
ENDIF(ARPACK_INCLUDE_DIRS)
set(ARPACK_INCLUDE_DIRS ${ARPACK_INCLUDE_DIR} )
set(ARPACK_LIBRARIES ${ARPACK_LIBRARY} )

SET(ARPACK_FOUND FALSE)
IF(ARPACK_INCLUDE_DIR AND ARPACK_LIBRARIES)
  MESSAGE(STATUS "ARPACK_INCLUDE_DIR=${ARPACK_INCLUDE_DIR}")
  MESSAGE(STATUS "ARPACK_LIBRARIES=${ARPACK_LIBRARIES}")
  SET(ARPACK_FOUND TRUE)
ENDIF()

MARK_AS_ADVANCED(
   ARPACK_INCLUDE_DIR
   ARPACK_LIBRARIES
   ARPACK_FOUND
)
