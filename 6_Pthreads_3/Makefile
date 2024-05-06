CC = g++
CFLAGS = -Wall -g -std=c++11 -pthread -fopenmp

build:
	@mkdir -p bin
	@mkdir -p output
	@mkdir -p build
	@$(CC) $(CFLAGS) src/heated_plate_openmp.c -o bin/heated_plate_openmp
	@$(CC) $(CFLAGS) src/parallel.cpp -shared -fPIC -o build/libparallel.so
	@$(CC) $(CFLAGS) src/heated_plate_pthreads.cpp -o bin/heated_plate_pthreads -Lbuild -lparallel

clean:
	@rm -rf bin
	@rm -rf build
	@echo "Clean complete"

run:
	@LD_LIBRARY_PATH=build bin/heated_plate_pthreads

runloop:
	@for np in 1 2 4 8 16; do \
		echo "Running with $$np threads"; \
		LD_LIBRARY_PATH=build bin/heated_plate_pthreads $$np; \
	done

reference:
	@bin/heated_plate_openmp

referenceloop:
	@for np in 1 2 4 8 16; do \
		echo "Running with $$np threads"; \
		bin/heated_plate_openmp $$np; \
	done