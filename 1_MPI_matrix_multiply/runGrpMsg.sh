echo "Running MPI matrix multiply using Group Message Passing" 
echo "Matrix_size | Number_of_Processes | Time" # | tee -a  ./output/GrpMsg.txt

for size in 128 256 512 1024 2048
# for size in 128 256
do
    for np in 1 2 4 8 
    do
        mpirun  -np $np ./bin/mpiGrpMsg $size | tee -a ./output/GrpMsg.txt
    done
    mpirun --use-hwthread-cpus -np 16 ./bin/mpiGrpMsg $size | tee -a ./output/GrpMsg.txt
done