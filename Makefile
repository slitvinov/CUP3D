.POSIX:
.SUFFIXES:
.SUFFIXES: .cpp
.SUFFIXES: .o

MPICXX = h5c++.mpich
GSL_CFLAGS != pkg-config --cflags gsl
GSL_LDFLAGS != pkg-config --libs gsl
CUBISMFLAGS = \
-DCUP_ALIGNMENT=64 \
-DCUP_BLOCK_SIZEX=8 \
-DCUP_BLOCK_SIZEY=8 \
-DCUP_BLOCK_SIZEZ=8 \
-DCUP_NO_MACROS_HEADER \
-DDIMENSION=3 \
-D_DOUBLE_PRECISION_ \
-DNDEBUG \
-I. \
-O3 \
-std=c++17 \

S = main.o
main: $(S:.cpp=.o)
	$(MPICXX) -o main $(S:.cpp=.o) $(GSL_LDFLAGS) $(LDFLAGS) -fopenmp
.cpp.o:
	$(MPICXX) -o $@ -c $< $(CUBISMFLAGS) $(CXXFLAGS) $(GSL_CFLAGS)
clean:
	-rm main $(S:.cpp=.o)
