CC = g++

CFLAGS = -g -Wall -std=c++11 -pthread

size = 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456

np = 1 2 4 8 16

# TARGET = bin/fomula

Fomula: src/Fomula.cpp
	@echo "Compiling Fomula..."
	@$(CC) $(CFLAGS) -o bin/Fomula src/Fomula.cpp
	@echo "Done!"
	@bin/Fomula

MonteCarlo: src/MonteCarlo.cpp
	@echo "Compiling MonteCarlo..."
	@$(CC) $(CFLAGS) -o bin/MonteCarlo src/MonteCarlo.cpp
	@echo "Done!"
	for size in $(size); do \
		for np in $(np); do \
			bin/MonteCarlo $$size $$np >> output/MonteCarlo.txt; \
		done \
	done

MonteCarloLoop: src/MonteCarlo.cpp
	@echo "Compiling MonteCarlo..."
	@$(CC) $(CFLAGS) -o bin/MonteCarlo src/MonteCarlo.cpp
	@echo "Done!"
	for round in 1 2 3 4 5 6 7 8 9 10; do \
		for size in $(size); do \
			for np in $(np); do \
				bin/MonteCarlo $$size $$np >> output/MonteCarlo.txt; \
			done \
		done \
	done

MonteCarloSingle: src/MonteCarlo.cpp
	@echo "Compiling MonteCarlo..."
	@$(CC) $(CFLAGS) -o bin/MonteCarlo src/MonteCarlo.cpp
	@echo "Done!"
	for round in 1 2 3 4 5 6 7 8 9 10; do \
		for size in $(size); do \
			bin/MonteCarlo $$size 1 >> output/loss.txt; \
		done \
	done

clean:
	@echo "Cleaning..."
	@rm -f bin/Fomula
	@rm -f bin/MonteCarlo
	@echo "Done!"