add_library(objlib OBJECT basis.cc ckpt.cc kpm.cc lanczos.cc lattice.cc miscellaneous.cc model.cc sparse.cc operators.cc)

target_compile_options(objlib     PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(objlib PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)

add_executable(test.x main_test.cc $<TARGET_OBJECTS:objlib>)

target_link_libraries(test.x ${LIBS} $<LINK_ONLY:MKL::MKL> OpenMP::OpenMP_CXX)
