CXX       = clang++
CPPFLAGS  = -O3 -std=c++11 -Wall -Wextra -Wno-objc-missing-property-synthesis -isystem /usr/local/Cellar/boost/1.62.0/include -m64 -isystem ${MKLROOT}/include -isystem ${HOME}/opt/arpackpp/include -isystem ${HOME}/opt/arpackpp/examples/areig
LIBS      = -L/usr/local/lib -larpack -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

OBJS      = measure.o lanczos.o sparse.o operators.o basis.o chain_xxz.o

OPTS      = 

EXEC      = ../bin/xxz.x


all: $(OBJS)
	$(CXX) $(OBJS) $(OPTS) $(LIBS) -o $(EXEC)

%.o: %.cc
	$(CXX) $(CPPFLAGS)  -c -o $@ $<

clean:
	$(RM) *.o *.a
