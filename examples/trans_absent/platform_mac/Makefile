CXX       = g++-7
CPPFLAGS  = -O2 -std=c++11 -m64 \
            -I ${HOME}/installs/include \
            -isystem ${MKLROOT}/include \
            -isystem ${HOME}/opt/arpackpp/include -isystem ${HOME}/opt/arpackpp/examples/areig
LIBS      = -L ${HOME}/installs/lib -lqbasis \
            -larpack \
            -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl \
            ${HOME}/installs/lib/libboost_filesystem.a ${HOME}/installs/lib/libboost_system.a

include ../object_list.txt

all: $(EXEC)
	@echo '************  COMPILED!  ************'

%.x: %.cc
	$(CXX) $(CPPFLAGS) $< $(LIBS) -o $@

clean:
	$(RM) *.o *.x
