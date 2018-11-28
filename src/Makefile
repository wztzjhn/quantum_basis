CXX       = g++-8
CPPFLAGS  = -O2 -std=c++11 -Wall -Wextra -m64 -fopenmp \
            -isystem ${MKLROOT}/include \
            -isystem ${HOME}/opt/arpackpp/include -isystem ${HOME}/opt/arpackpp/examples/areig \
            -isystem ${HOME}/installs/include
LIBS      = -larpack \
            -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl \
            -fopenmp \
            ${HOME}/installs/lib/libboost_filesystem.a ${HOME}/installs/lib/libboost_system.a

#comment: The default boost libs on mac are typically built with clang, incompatible with g++. 
#         To resolve, we built a separate boost with g++, in folder ${HOME}/installs

include makefile_list/object_list.txt

libqbasis.a: $(OBJS)
	ar rvs libqbasis.a $(OBJS)

install: libqbasis.a
	$(RM) ${HOME}/installs/lib/libqbasis.*; cp libqbasis.a ${HOME}/installs/lib; cp qbasis.h ${HOME}/installs/include

%.o: %.cc
	$(CXX) $(CPPFLAGS)  -c -o $@ $<

clean:
	$(RM) *.o *.a
