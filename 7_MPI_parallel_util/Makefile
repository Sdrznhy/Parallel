CC = g++
CFLAGS = -Wall -g -std=c++11 -pthread -fopenmp

build:
	@mkdir -p bin
	@$(CC) $(CFLAGS) src/fft_serial.cpp -o bin/fft_serial
	@mpic++ $(CFLAGS) src/fft_parallel.cpp -o bin/fft_parallel

clean:
	@rm -rf bin
	@echo "Clean complete"

runSerial:
	@bin/fft_serial

runParallel:
	@echo "fft_PARALLEL"
	@echo "  C++ version"
	@echo "  Demonstrate an implementation of the Fast Fourier Transform"
	@echo "  of a complex data vector."
	@echo "  Accuracy check:"
	@echo "    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)"
	@echo "             N      NITS    Error         Time          Time/Call     MFLOPS"

	@ for ln2 in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do \
		# ./bin/fft_parallel $$ln2; \
		mpirun --use-hwthread-cpus -np 8 bin/fft_parallel $$ln2; \
	done