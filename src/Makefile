CXX       = g++
CPPFLAGS  = -O2 -std=c++11 -Wall -Wextra -m64 -fopenmp \
            -isystem ${MKLROOT}/include \
            -isystem ${HOME}/opt/arpackpp/include -isystem ${HOME}/opt/arpackpp/examples/areig
LIBS      = -larpack \
            -L${MKLROOT}/lib/intel64 -L${MKLROOT}/../../compiler/latest/linux/compiler/lib/intel64_lin -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl \
            -fopenmp \
            -lboost_filesystem -lboost_system

include makefile_list/object_list.txt

libqbasis.a: $(OBJS)
	ar rvs libqbasis.a $(OBJS)

install: libqbasis.a
	$(RM) ${HOME}/installs/lib/libqbasis.*; cp libqbasis.a ${HOME}/installs/lib; cp qbasis.h ${HOME}/installs/include

%.o: %.cc
	$(CXX) $(CPPFLAGS) -c -o $@ $<

clean:
	$(RM) *.o *.a
