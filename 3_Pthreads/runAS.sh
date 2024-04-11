echo "running array sum up using pthread"
echo "array_size(M)|thread_num|time(ms)"

# for round in {1..10}
# do
    for size in 1 2 4 8 16 32 64 128
    do
        for np in 1 2 4 8 16
        do
            ./bin/ArraySum_GPT  $size $np >> output/Array.txt
            # ./bin/ArraySum $size $np >> output/Array.txt
        done
    done
# done
