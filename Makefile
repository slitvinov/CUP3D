.POSIX:
MPICC = mpicc
CFLAGS = -O3
TAB = lab_ss1_t1.bin lab_ss1_t0.bin lab_ss2_t1.bin lab_ss3_t0.bin
all: main $(TAB)
main: main.c
	$(MPICC) -o main main.c -std=c99 -D_GNU_SOURCE -fopenmp $(CFLAGS) $(LDFLAGS) -lm
$(TAB): gen_table.py
	python3 gen_table.py
clean:
	-rm main $(TAB)
