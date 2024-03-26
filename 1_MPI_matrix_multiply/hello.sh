# sudo chmod u+x hello.sh

mpic++ -o ./bin/mpihello ./src/mpihello.cpp
mpirun -np 4 ./bin/mpihello
