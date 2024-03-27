echo "Running 1_MPI_matrix_multiply" 
echo "Matrix_size | Number_of_Processes | Time" | tee -a  ./output/1_MPI_matrix_multiply.txt

for size in 128 256 512 1024 2048
do
    for np in 1 2 4 8 
    do
        mpirun  -np $np ./bin/mpiP2P $size | tee -a ./output/1_MPI_matrix_multiply.txt
    done
    mpirun --use-hwthread-cpus -np 16 ./bin/mpiP2P $size | tee -a ./output/1_MPI_matrix_multiply.txt
done