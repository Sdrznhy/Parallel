// #include <cmath>
// #include <cstdlib>
// #include <ctime>
// #include <iomanip>
// #include <iostream>
#include <bits/stdc++.h>
#include <mpi.h>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[]);
double cpu_time(void);
double ggl(double* ds);
void ccopy(int n, double x[], double y[]);
void cfft2(int n, double x[], double y[], double w[], double sgn);
void cffti(int n, double w[]);
void step(int n, int mj, double a[], double b[], double c[], double d[],
    double w[], double sgn);
void timestamp();

int main(int argc, char* argv[])
{
    // 声明变量
    double ctime, ctime1, ctime2;
    double error;
    // int first; //?
    double flops, mflops;
    double fnm1; // ?
    // int i, icase, it; // ?
    int i, it;
    int ln2;
    int n; // ?
    int nits = 10000;
    static double seed;
    double sgn;
    double *w, *x, *y, *z;
    double z0, z1;

    // timestamp();

    seed = 331.0;
    n = 1; // ？

    ln2 = argc > 1 ? atoi(argv[1]) : 10;

    for (i = 1; i < ln2; i++) {
        if ((i % 4) == 0) {
            nits = nits / 10;
        }
        if (nits < 1) {
            nits = 1;
        }
    }

    // for (ln2 = 1; ln2 <= 20; ln2++) {
    n = pow(2, ln2);
    // n = 2 * n;
    // first

    w = new double[n];
    x = new double[2 * n];
    y = new double[2 * n];
    z = new double[2 * n];

    for (i = 0; i < 2 * n; i = i + 2) {
        z0 = ggl(&seed);
        z1 = ggl(&seed);
        x[i] = z0;
        z[i] = z0;
        x[i + 1] = z1;
        z[i + 1] = z1;
    }

    cffti(n, w);

    MPI_Init(&argc, &argv);
    int rank, thread_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &thread_num);

    sgn = +1.0;
    cfft2(n, x, y, w, sgn);
    sgn = -1.0;
    cfft2(n, y, x, w, sgn);

    if (rank == 0) {
        fnm1 = 1.0 / (double)n;
        error = 0.0;
        for (i = 0; i < 2 * n; i = i + 2) {
            error = error
                + pow(z[i] - fnm1 * x[i], 2)
                + pow(z[i + 1] - fnm1 * x[i + 1], 2);
        }
        error = sqrt(fnm1 * error);
        cout << "  " << setw(12) << n
             << "  " << setw(8) << nits
             << "  " << setw(12) << error;
    }
    // not first
    for (i = 0; i < 2 * n; i = i + 2) {
        z0 = 0.0;
        z1 = 0.0;
        x[i] = z0;
        z[i] = z0;
        x[i + 1] = z1;
        z[i + 1] = z1;
    }

    cffti(n, w);

    // MPI_Init(&argc, &argv);

    // **************************************
    ctime1 = cpu_time();
    for (it = 0; it < nits; it++) {
        sgn = +1.0;
        cfft2(n, x, y, w, sgn);
        sgn = -1.0;
        cfft2(n, y, x, w, sgn);
    }
    ctime2 = cpu_time();
    ctime = ctime2 - ctime1;
    // **************************************

    MPI_Finalize();

    if (rank == 0) {
        flops = 2.0 * (double)nits * (5.0 * (double)n * (double)ln2);

        mflops = flops / 1.0E+06 / ctime;

        cout << "  " << setw(12) << ctime
             << "  " << setw(12) << ctime / (double)(2 * nits)
             << "  " << setw(12) << mflops << "\n";

        // if ((ln2 % 4) == 0) {
        //     nits = nits / 10;
        // }
        // if (nits < 1) {
        //     nits = 1;
        // }
        delete[] w;
        delete[] x;
        delete[] y;
        delete[] z;
    }
    // }

    // cout << "\n";
    // cout << "FFT_PARALLEL:\n";
    // cout << "  Normal end of execution.\n";
    // cout << "\n";
    // timestamp();

    return 0;
}

double cpu_time(void)
{
    double value;
    value = (double)clock() / (double)CLOCKS_PER_SEC;
    return value;
}

void timestamp()
{
#define TIME_SIZE 40

    static char time_buffer[TIME_SIZE];
    const struct tm* tm;
    time_t now;

    now = time(NULL);
    tm = localtime(&now);

    strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

    cout << time_buffer << "\n";

    return;
#undef TIME_SIZE
}

void ccopy(int n, double x[], double y[])
{
    int i;

    for (i = 0; i < n; i++) {
        y[i * 2 + 0] = x[i * 2 + 0];
        y[i * 2 + 1] = x[i * 2 + 1];
    }
    return;
}

double ggl(double* seed)
{
    double d2 = 0.2147483647e10;
    double t;
    double value;

    t = *seed;
    t = fmod(16807.0 * t, d2);
    *seed = t;
    value = (t - 1.0) / (d2 - 1.0);

    return value;
}

void cfft2(int n, double x[], double y[], double w[], double sgn)
{
    int j;
    int m;
    int mj;
    int tgle;

    m = (int)(log((double)n) / log(1.99));
    mj = 1;
    //
    //  Toggling switch for work array.
    //
    tgle = 1;
    step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

    if (n == 2) {
        return;
    }

    for (j = 0; j < m - 2; j++) {
        mj = mj * 2;
        if (tgle) {
            step(n, mj, &y[0 * 2 + 0], &y[(n / 2) * 2 + 0], &x[0 * 2 + 0], &x[mj * 2 + 0], w, sgn);
            tgle = 0;
        } else {
            step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);
            tgle = 1;
        }
    }
    //
    //  Last pass thru data: move y to x if needed
    //
    if (tgle) {
        ccopy(n, y, x);
    }

    mj = n / 2;
    step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

    return;
}

void cffti(int n, double w[])
{
    double arg;
    double aw;
    int i;
    int n2;
    const double pi = 3.141592653589793;

    n2 = n / 2;
    aw = 2.0 * pi / ((double)n);

    for (i = 0; i < n2; i++) {
        arg = aw * ((double)i);
        w[i * 2 + 0] = cos(arg);
        w[i * 2 + 1] = sin(arg);
    }
    return;
}

void step(int n, int mj, double a[], double b[], double c[],
    double d[], double w[], double sgn)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double ambr, ambu;
    int j, ja, jb, jc, jd, jw, k, lj, mj2;
    double wjw[2];

    mj2 = 2 * mj;
    lj = n / mj2;

    // Calculate chunk and remainder
    int chunk = lj / size;
    int remainder = lj % size;

    int start = chunk * rank;
    int end = start + chunk + (rank == size - 1 ? remainder : 0);
    int local_n = end - start;

    // chunk * mj = n / mj2 / size * mj = n / size / 2

    // Create local arrays for c and d
    double* local_c = new double[local_n * mj];
    double* local_d = new double[local_n * mj];

    for (j = start; j < end; j++) {
        jw = j * mj;
        ja = jw;
        jb = ja;
        jc = j * mj2;
        jd = jc;

        wjw[0] = w[jw * 2 + 0];
        wjw[1] = w[jw * 2 + 1];

        if (sgn < 0.0) {
            wjw[1] = -wjw[1];
        }

        for (k = 0; k < mj; k++) {
            c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0];
            c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1];

            ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0];
            ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1];

            d[(jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu;
            d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu;
        }
    }

    for (int i = 0; i < local_n * mj; i++) {
        local_c[i] = c[start * mj2 + i];
        local_d[i] = d[start * mj2 + i];
    }

    // Calculate send counts and displacements for each process
    int* sendcounts = new int[size];
    int* displs = new int[size];
    for (int i = 0; i < size; i++) {
        sendcounts[i] = chunk * mj + (i == size - 1 ? remainder * mj : 0);
        displs[i] = i * chunk * mj;
    }

    // Use MPI_Gatherv to collect the results from all processes
    MPI_Gatherv(local_c, local_n * mj, MPI_DOUBLE, c, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_d, local_n * mj, MPI_DOUBLE, d, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[] sendcounts;
    delete[] displs;
    delete[] local_c;
    delete[] local_d;
}
