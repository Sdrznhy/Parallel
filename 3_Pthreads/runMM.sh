echo "running matrix multiplication using pthread"
echo "matrix_size|thread_num|time used(s)"
# for size in 128 256 512 1024 2048
# do
#     for np in 1 2 4 8 16
#     do
#         ./bin/MatrixMultiply $size $np >> ./output/Matrix.txt
#         # ./bin/MatrixMultiply_GPT $size $np >> ./output/Matrix.txt
#     done
# done
for round in {1..20}
do
    for size in 128 256 512 1024 2048
    do
        for np in 1 2 4 8 16
        do
            ./bin/MatrixMultiply_GPT $size $np >> ./output/Matrix.txt
        done
    done
done