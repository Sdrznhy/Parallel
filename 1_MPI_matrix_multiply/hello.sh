# sudo chmod u+x hello.sh

mpic++ -o ./bin/mpihello ./src/mpihello.cpp
# mpirun -np  8 ./bin/mpihello
mpirun --use-hwthread-cpus -np 16 ./bin/mpihello
