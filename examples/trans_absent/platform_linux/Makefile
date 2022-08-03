CXX       = g++
CXXFLAGS  = -std=c++17 -Wall -O3 -funroll-loops -m64 -march=native -fopenmp -I${MKL_INC_DIR} \
            -I ${HOME}/installs/include
LIBS      = -L ${HOME}/installs/lib -lqbasis \
            -larpack \
            -L${MKL_LIB_DIR} -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_tbb_thread -lmkl_core -lpthread -lm -ldl

include ../object_list.txt

all: $(EXEC)
	@echo '************  COMPILED!  ************'
%.x: %.cc
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@
clean:
	$(RM) *.o *.x
