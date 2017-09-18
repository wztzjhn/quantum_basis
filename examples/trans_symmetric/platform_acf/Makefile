CXX       = icpc
CPPFLAGS  = -O2 -std=c++11 -traceback -DMKL_ILP64 \
            -I ${HOME}/installs/include \
            -isystem ${MKLROOT}/include \
            -isystem ${HOME}/opt/arpackpp/include -isystem ${HOME}/opt/arpackpp/examples/areig
LIBS      = -L ${HOME}/installs/lib -lqbasis \
            -L${HOME}/opt/ARPACK_ILP64 -larpack -lgfortran \
            -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl \
            -L /sw/acf/boost/1.64.0/centos7.3_binary/lib -lboost_filesystem -lboost_system

include ../object_list.txt

all: $(EXEC)
	@echo '************  COMPILED!  ************'

%.x: %.cc
	$(CXX) $(CPPFLAGS) $< $(LIBS) -o $@

clean:
	$(RM) *.o *.x
