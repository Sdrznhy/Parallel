build:
	@rm -rf bin
	@mkdir bin
	@nvcc src/Hello.cu -o bin/Hello
	@nvcc src/MatrixTranspose.cu -o bin/MatrixTranspose

clean:
	@rm -rf bin

runHello:
	@bin/Hello

runMatrixTranspose:
	@for size in 128 256 512 1024 2048 4096; do \
		for n in 16 32 64 128 256; do \
			bin/MatrixTranspose $$size $$n $$n; \
		done; \
		# bin/MatrixTranspose $$size; \
	done