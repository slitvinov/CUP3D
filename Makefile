.POSIX:
MPICC = mpicc
CFLAGS = -O3
TAB = lab_ss1_t1.bin lab_ss1_t0.bin lab_ss2_t1.bin lab_ss3_t0.bin
all: main $(TAB)
main: main.c
	$(MPICC) -o main main.c -std=c99 -D_GNU_SOURCE -fopenmp $(CFLAGS) $(LDFLAGS) -lm
$(TAB): gen_table.py
	python3 gen_table.py
check: main.c
	$(MPICC) -fanalyzer -Wall -Wextra -O3 -std=c99 -D_GNU_SOURCE -fopenmp -fsyntax-only main.c
	cppcheck --language=c -U__cplusplus --enable=warning,performance,portability --std=c99 -q main.c
asan: main.c
	OMPI_CC=/opt/homebrew/opt/llvm/bin/clang $(MPICC) -o main_asan main.c -std=c99 -D_GNU_SOURCE -fopenmp -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -lm
clean:
	-rm main $(TAB)
