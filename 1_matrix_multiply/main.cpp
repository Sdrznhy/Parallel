# include <iostream>
# include <cstdlib> // for rand() and srand()
# include <ctime> // for time()

int main() {
    srand(time(0)); // use current time as seed for random generator

    int m = 777, n = 787, k = 747;
    int A[m][n], B[n][k];

    // Generate m x n matrix A
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            A[i][j] = rand() ; // Generate random number
        }
    }

    // Generate n x k matrix B
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < k; j++) {
            B[i][j] = rand() ; // Generate random number
        }
    }

    // Multiply A and B to get C
    int C[m][k];
    // start time
    clock_t start = clock();
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < k; j++) {
            C[i][j] = 0;
            for(int l = 0; l < n; l++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
    // end time
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout << "function: " << "original" << std::endl;
    std::cout << "Time taken: " << time << " seconds" << std::endl;
}