import time
import numpy as np

m = 1024
n = 1024
k = 1024

if __name__ == "__main__":
    start = time.time()
    # 生成m*n的矩阵A和n*k的矩阵B
    A = np.random.rand(m, n)
    B = np.random.rand(n, k)
    C = np.dot(A, B)
    end = time.time()
    # 保留小数点后两位
    print("m={}, n={}, k={}".format(m, n, k))
    print("using numpy:")
    print("Time: {:.2f}ms".format((end - start) * 1000))

    start = time.time()
    C = np.zeros((m, k))
    for i in range(m):
        for j in range(k):
            for l in range(n):
                C[i][j] += A[i][l] * B[l][j]
    end = time.time()
    print("using original python:")
    print("Time: {:.2f}ms".format((end - start) * 1000))
