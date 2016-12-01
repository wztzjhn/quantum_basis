add_executable(test.x lanczos.cc sparse.cc operators.cc basis.cc main.cc)
target_link_libraries(test.x ${LIBS})

