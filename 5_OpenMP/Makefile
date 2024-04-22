CC = g++
CFLAGS = -g -Wall -std=c++11 -pthread
size = 128 256 512 1024 2048
np = 1 2 4 8 16

MatrixMultiply: src/MatrixMultiply.cpp
	@echo "Compiling MatrixMultiply..."
	@$(CC) $(CFLAGS) -o bin/MatrixMultiply src/MatrixMultiply.cpp
	@echo "Done!"
	@for size in $(size); do \
		for np in $(np); do \
			bin/MatrixMultiply $$size $$np >> output/MatrixMultiply.txt; \
		done \
	done
