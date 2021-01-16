import numpy as np
from scipy.sparse import csr_matrix


def L_U(matrix):
    matrix = matrix.A
    n = matrix.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i <= j:
                U[i][j] = matrix[i][j] - get_sum_i(L, U, i, j)
            else:
                L[i][j] = (matrix[i][j] - get_sum_j(L, U, i, j)) / U[j][j]
    return csr_matrix(L), csr_matrix(U)


def get_sum_i(L, U, i, j):
    ans = 0
    for k in range(0, i):
        ans += L[i][k] * U[k][j]
    return ans


def get_sum_j(L, U, i, j):
    ans = 0
    for k in range(0, j):
        ans += L[i][k] * U[k][j]
    return ans

