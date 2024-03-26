echo "Running 1_MPI_matrix_multiply" 
echo "Matrix_size | Number_of_Processes | Time" | tee ./output/1_MPI_matrix_multiply.txt
mpirun --use-hwthread-cpus -np 1 ./bin/main 512 | tee -a ./output/1_MPI_matrix_multiply.txt
mpirun --use-hwthread-cpus -np 2 ./bin/main 512 | tee -a ./output/1_MPI_matrix_multiply.txt
mpirun --use-hwthread-cpus -np 4 ./bin/main 512 | tee -a ./output/1_MPI_matrix_multiply.txt
mpirun --use-hwthread-cpus -np 8 ./bin/main 512 | tee -a ./output/1_MPI_matrix_multiply.txt
mpirun --use-hwthread-cpus -np 16 ./bin/main 512 | tee -a ./output/1_MPI_matrix_multiply.txt