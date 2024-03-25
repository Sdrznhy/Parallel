# chmod u+x hello.sh
mpicc -o ./bin/hello hello.c
mpirun -np 4 ./bin/hello