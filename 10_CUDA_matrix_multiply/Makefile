
build:
	@rm -rf bin
	@mkdir bin
	@nvcc src/shared_memory.cu -o bin/shared_memory
	@nvcc src/origin.cu -o bin/origin
	@nvcc src/limit.cu -o bin/limit

clean:
	@rm -rf bin

shared_memory:
	@for block_size in 8 16 32; do \
		bin/shared_memory 128 $$block_size $$block_size; \
		bin/shared_memory 256 $$block_size $$block_size; \
		bin/shared_memory 512 $$block_size $$block_size; \
		bin/shared_memory 1024 $$block_size $$block_size; \
		bin/shared_memory 2048 $$block_size $$block_size; \
		bin/shared_memory 4096 $$block_size $$block_size; \
		bin/shared_memory 8192 $$block_size $$block_size; \
	done


origin:
	@for block_size in 8 16 32; do \
		bin/origin 128 $$block_size $$block_size; \
		bin/origin 256 $$block_size $$block_size; \
		bin/origin 512 $$block_size $$block_size; \
		bin/origin 1024 $$block_size $$block_size; \
		bin/origin 2048 $$block_size $$block_size; \
		bin/origin 4096 $$block_size $$block_size; \
		bin/origin 8192 $$block_size $$block_size; \
	done

limit:
	@bin/limit
	