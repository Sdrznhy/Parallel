echo "running array sum up using pthread"
echo "array_size(M)|thread_num|time(us)"

for size in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
do
    for np in 1 2 4 8 16
    do
        # ./bin/ArraySum $np $size | tee -a output/Array2.txt
        ./bin/ArraySum $np $size >> output/Array.txt
    done
done
