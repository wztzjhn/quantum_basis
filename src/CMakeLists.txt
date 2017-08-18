set(QBASIS_SRC
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

#add_executable(Heisenberg_spin_half_triangular.x ../examples/trans_symmetric/Heisenberg_spin_half_triangular.cc ${QBASIS_SRC})
#target_link_libraries(Heisenberg_spin_half_triangular.x ${LIBS})

add_executable(test.x test.cc test_basis.cc test_operator.cc test_lanczos.cc test_lattice.cc ${QBASIS_SRC})
target_link_libraries(test.x ${LIBS})

add_executable(ss_energy.x ../../2017_ShastrySutherland/variational/src/subroutines.cc ../../2017_ShastrySutherland/variational/src/ss_energy.cc ${QBASIS_SRC})
target_link_libraries(ss_energy.x ${LIBS})

add_executable(ss_chi.x ../../2017_ShastrySutherland/variational/src/subroutines.cc ../../2017_ShastrySutherland/variational/src/ss_chi.cc ${QBASIS_SRC})
target_link_libraries(ss_chi.x ${LIBS})

#add_executable(ex.x ../../2017_Filip/lanczos/src/exchange.cc ${QBASIS_SRC})
#target_link_libraries(ex.x ${LIBS})

set_target_properties(qbasis PROPERTIES PUBLIC_HEADER "${QBASIS_INCLUDE}")

# Add the qbasis library and headers to the "export-set"
install(TARGETS qbasis
  EXPORT qbasisTargets
  LIBRARY       DESTINATION "${INSTALL_LIB_DIR}"     COMPONENT shlib
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT dev)

