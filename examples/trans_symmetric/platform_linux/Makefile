CXX       = g++
CXXFLAGS  = -std=c++17 -Wall -O3 -funroll-loops -m64 -DMKL_ILP64 -march=native \
            -I${HOME}/installs/include -I${MKL_INC_DIR}
LIBS      = -L${HOME}/installs/lib -lqbasis -larpackILP64 \
            -L${MKL_LIB_DIR} -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lpthread -ltbb -lm -ldl


include ../object_list.txt

all: $(EXEC)
	@echo '************  COMPILED!  ************'
%.x: %.cc
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@
clean:
	$(RM) *.o *.x *.txt *.dat
