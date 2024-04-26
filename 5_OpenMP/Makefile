CC = g++
CFLAGS = -g -Wall -std=c++11 -pthread
size = 128 256 512 1024 2048
np = 1 2 4 8 16

MatrixMultiply: src/MatrixMultiply.cpp
	@for size in $(size); do \
		for np in $(np); do \
			bin/MatrixMultiply $$size $$np;  \
		done \
	done

PthreadFor: src/MM_PthreadFor.cpp
	@for size in $(size); do \
		for np in $(np); do \
			LD_LIBRARY_PATH=build bin/MM_PthreadFor $$size $$np >> output/MM_PthreadFor.txt; \
		done \
	done


build:
	@mkdir -p bin
	@mkdir -p output
	@mkdir -p build
	@$(CC) $(CFLAGS) src/MatrixMultiply.cpp -fopenmp -o bin/MatrixMultiply
	@$(CC) $(CFLAGS) src/parallel.cpp -shared -fPIC -o build/libparallel.so
	@$(CC) $(CFLAGS) src/MM_PthreadFor.cpp -o bin/MM_PthreadFor -L build -l parallel
	@echo "Build complete"

clean:
	@rm -rf bin
	@# @rm -rf output
	@rm -rf build
	@echo "Clean complete"
	
