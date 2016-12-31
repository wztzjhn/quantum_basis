set(QBASIS_SRC
  measure.cc
  lattice.cc
  model.cc
  lanczos.cc
  sparse.cc
  operators.cc
  basis.cc
  miscellaneous.cc
)

set(QBASIS_INCLUDE qbasis.h)

add_library(qbasis SHARED ${QBASIS_SRC})
target_link_libraries(qbasis ${LIBS})

add_executable(test.x test.cc ${QBASIS_SRC})
target_link_libraries(test.x ${LIBS})

add_executable(xxz.x chain_xxz.cc ${QBASIS_SRC})
target_link_libraries(xxz.x ${LIBS})

add_executable(ss.x examples/shastry_sutherland.cc ${QBASIS_SRC})
target_link_libraries(ss.x ${LIBS})

set_target_properties(qbasis PROPERTIES PUBLIC_HEADER "${QBASIS_INCLUDE}")

# Add the qbasis library and headers to the "export-set"
install(TARGETS qbasis
  EXPORT qbasisTargets
  LIBRARY       DESTINATION "${INSTALL_LIB_DIR}"     COMPONENT shlib
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT dev)

