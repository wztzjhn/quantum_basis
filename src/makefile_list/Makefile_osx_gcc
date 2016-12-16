CXX       = g++-5
CPPFLAGS  = -O2 -std=c++11 -Wall -Wextra -Wpadded -isystem /usr/local/Cellar/boost/1.62.0/include -m64 -isystem ${MKLROOT}/include -isystem ${HOME}/opt/arpackpp/include -isystem ${HOME}/opt/arpackpp/examples/areig
LIBS      = -L/usr/local/lib -larpack -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

OBJS      = measure.o lanczos.o sparse.o operators.o basis.o test.o

OPTS      = 

EXEC      = ../bin/test.x


all: $(OBJS)
	$(CXX) $(OBJS) $(OPTS) $(LIBS) -o $(EXEC)

%.o: %.cc
	$(CXX) $(CPPFLAGS)  -c -o $@ $<

clean:
	$(RM) *.o
