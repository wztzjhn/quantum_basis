CXX       = g++
CXXFLAGS  = -std=c++17 -Wall -O3 -funroll-loops -m64 -march=native -fPIC -fopenmp -I${MKL_INC_DIR} \
            -isystem ${HOME}/opt/arpackpp/include -isystem ${HOME}/opt/arpackpp/examples/areig
LIBS      = -larpack \
            -L${MKL_LIB_DIR} -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl \
            -fopenmp
OBJS = basis.o ckpt.o lanczos.o lattice.o miscellaneous.o model.o operators.o sparse.o


test: main_test.o $(OBJS)
	$(CXX) main_test.o $(OBJS) $(LIBS) -o ../bin/test.x
libqbasis.so: $(OBJS)
	$(CXX) -shared -fPIC $(OBJS) $(LIBS) -o libqbasis.so
install: libqbasis.so
	cp libqbasis.so ${HOME}/installs/lib; cp qbasis.h ${HOME}/installs/include
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<
clean:
	$(RM) *.o *.a *.so *.dylib
