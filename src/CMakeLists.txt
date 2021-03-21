set(QBASIS_SRC
  basis.cc
  ckpt.cc
  kpm.cc
  lanczos.cc
  lattice.cc
  miscellaneous.cc
  model.cc
  sparse.cc
  operators.cc
)

set(QBASIS_INCLUDE qbasis.h)

add_library(qbasis SHARED ${QBASIS_SRC})
target_link_libraries(qbasis ${LIBS})

add_executable(half_repr.x ../examples/trans_symmetric/latt_chain/chain_Heisenberg_spin_half.cc ${QBASIS_SRC})
target_link_libraries(half_repr.x PUBLIC ${LIBS} OpenMP::OpenMP_CXX)

add_executable(playground.x ../examples/trans_absent/playground.cc ${QBASIS_SRC})
target_link_libraries(playground.x PUBLIC ${LIBS} OpenMP::OpenMP_CXX)

set_target_properties(qbasis PROPERTIES PUBLIC_HEADER "${QBASIS_INCLUDE}")

# Add the qbasis library and headers to the "export-set"
install(TARGETS qbasis
  EXPORT qbasisTargets
  LIBRARY       DESTINATION "${INSTALL_LIB_DIR}"     COMPONENT shlib
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT dev)

