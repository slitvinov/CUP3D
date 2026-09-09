.POSIX:
MPICC = mpicc
CFLAGS = -O3
main: main.c
	$(MPICC) -o main main.c -std=c99 -fopenmp $(CFLAGS) $(LDFLAGS) -lm
clean:
	-rm main
