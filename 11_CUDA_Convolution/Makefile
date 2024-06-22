
build:
	@rm -rf bin
	@mkdir bin
	@nvcc -o bin/slide src/slide.cu
	@nvcc -o bin/im2col src/im2col.cu
	@nvcc -o bin/cudnn src/cudnn.cu -lcudnn

slide: 
	@./bin/slide 512 512

im2col:
	@./bin/im2col 512 512

cudnn:
	@./bin/cudnn 512 512