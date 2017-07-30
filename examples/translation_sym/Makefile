CXX       = g++-7
CPPFLAGS  = -O2 -std=c++11 -I ${HOME}/installs/include -m64 -isystem ${MKLROOT}/include -isystem ${HOME}/opt/arpackpp/include -isystem ${HOME}/opt/arpackpp/examples/areig
LIBS      = -L ${HOME}/installs/lib -lqbasis -L/usr/local/lib -larpack -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -fopenmp

EXEC      = Heisenberg_spin_half_chain.x
#            Heisenberg_spin_half_triangular.x \
#            Fermi_Hubbard_square.x \
#            Kondo_chain.x

all: $(EXEC)
	@echo '************  COMPILED!  ************'

%.x: %.cc
	$(CXX) $(CPPFLAGS) $< $(LIBS) -o $@

clean:
	$(RM) *.o *.x
