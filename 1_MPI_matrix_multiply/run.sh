echo "Running 1_MPI_matrix_multiply" 
echo "Matrix_size | Number_of_Processes | Time" | tee  ./output/1_MPI_matrix_multiply.txt

for size in 128 256 512 1024 2048
do
    for np in 1 2 4 8 16
    do
        mpirun --use-hwthread-cpus -np $np ./bin/main $size | tee -a ./output/1_MPI_matrix_multiply.txt
    done
done