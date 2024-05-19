CC = g++
CFLAGS = -Wall -g -std=c++11 -pthread -fopenmp

build:
	@rm -rf bin
	@mkdir -p bin
	@$(CC) $(CFLAGS) src/floyd_serial.cpp -o bin/floyd_serial
	@$(CC) $(CFLAGS) src/floyd_parallel.cpp -o bin/floyd_parallel
	@echo "Build complete"

clean:
	@rm -rf bin
	@echo "Clean complete"

runSerial:
	@bin/floyd_serial

runParallel:
	@bin/floyd_parallel