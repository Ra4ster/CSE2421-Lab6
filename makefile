CFLAGS = -O3 -g -march=native
LIBS = -lopenblas -lm -mavx2 -mfma
SRC = Main.c MatrixMath.c

all: clang # Expected


# GCC build
gcc:
	@gcc $(CFLAGS) $(SRC) -o ./out/gccMM_5000x5000 $(LIBS)

# Clang build
clang:
	@clang $(CFLAGS) $(SRC) -o ./out/clangMM_5000x5000 $(LIBS)
