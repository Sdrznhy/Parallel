
build:
	@rm -rf bin
	@mkdir bin
	@nvcc -o bin/slide src/slide.cu
	@nvcc -o bin/im2col src/im2col.cu
	@nvcc -o bin/cudnn src/cudnn.cu -lcudnn

slide: 
	@./bin/slide 256 256
	@./bin/slide 512 512
	@./bin/slide 1024 1024
	@./bin/slide 2048 2048

im2col:
	@./bin/im2col 256 256
	@./bin/im2col 512 512
	@./bin/im2col 1024 1024
	@./bin/im2col 2048 2048

cudnn:
	@./bin/cudnn 256 256
	@./bin/cudnn 512 512
	@./bin/cudnn 1024 1024
	@./bin/cudnn 2048 2048