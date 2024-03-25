
#include <stdio.h>
#include <mpi/mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    printf("Hello from rank %d of %d\n", world_rank, world_size);
    MPI_Finalize();
}