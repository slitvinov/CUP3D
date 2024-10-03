.POSIX:
.SUFFIXES:
.SUFFIXES: .cpp
.SUFFIXES: .o

MPICXX = mpicxx
GSL_CFLAGS != pkg-config --cflags gsl
GSL_LDFLAGS != pkg-config --libs gsl
FLAGS = \
-DCUBISM_ALIGNMENT=64 \
-D_BS_=8 \
-DDIMENSION=3 \
-DNDEBUG \
-O3 \
-std=c++17 \

S = main.o
main: $(S:.cpp=.o)
	$(MPICXX) -o main $(S:.cpp=.o) $(GSL_LDFLAGS) $(LDFLAGS) -fopenmp
.cpp.o:
	$(MPICXX) -o $@ -c $< $(FLAGS) $(CXXFLAGS) $(GSL_CFLAGS)
clean:
	-rm main $(S:.cpp=.o)
