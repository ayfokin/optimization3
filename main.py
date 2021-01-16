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


def gauss_method(A, F):
    length = A.shape[0]
    L, U = L_U(A)
    y = [0 for i in range(length)]
    for i in range(length):
        tmp = F[i]
        for j in range(L.indptr[i], L.indptr[i + 1]):
            index = L.indices[j]
            l_j = L.data[j]
            tmp -= y[index] * l_j
            if j == L.indptr[i + 1] - 1:
                y[index] = tmp / l_j
    x = [0 for i in range(length)]
    length = len(U.indptr) - 1
    for i in range(length, 0, -1):
        tmp = y[i - 1]
        for j in range(U.indptr[i] - 1, U.indptr[i - 1] - 1, -1):
            index = U.indices[j]
            u_j = U.data[j]
            tmp -= x[index] * u_j
            if j == U.indptr[i - 1]:
                x[index] = tmp / u_j
    return x


a = csr_matrix(np.array([[10, 6, 2, 0], [5, 1, -2, 4], [3, 5, 1, -1], [0, 6, -2, 2]]))
L, U = L_U(a)
print(L.A)
print(U.A)
print(a)
#Тестовое уравнение
A = csr_matrix(np.array([[1, 0, 2], [3, -1, 0], [4, -1, 3]]))
F = [1, 1, 1]
L, U = L_U(A)
print(L.A)
print(U.A)
print(A)
print(gauss_method(A, F))
