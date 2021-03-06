import numpy as np
import random
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
    L, U = L_U(A)
    y = up_to_down_gauss(L, F)
    x = down_to_up_gauss(U, y)
    return x


def up_to_down_gauss(initial, result):
    length = initial.shape[0]
    x = [0 for i in range(length)]
    for i in range(length):
        tmp = result[i]
        for j in range(initial.indptr[i], initial.indptr[i + 1]):
            index = initial.indices[j]
            i_j = initial.data[j]
            tmp -= x[index] * i_j
            if j == initial.indptr[i + 1] - 1:
                x[index] = tmp / i_j
    return x


def down_to_up_gauss(initial, result):
    length = initial.shape[0]
    x = [0 for i in range(length)]
    for i in range(length, 0, -1):
        tmp = result[i - 1]
        for j in range(initial.indptr[i] - 1, initial.indptr[i - 1] - 1, -1):
            index = initial.indices[j]
            i_j = initial.data[j]
            tmp -= x[index] * i_j
            if j == initial.indptr[i - 1]:
                x[index] = tmp / i_j
    return x


def create_gilbert_matrix(k):
    a = []
    for i in range(k):
        a.append([])
        for j in range(k):
            a[i].append(1 / (i + j + 1))
    return a


def create_matrix(k):
    a = []
    for i in range(k):
        a.append([])
        for j in range(k):
            element = random.randrange(-4, 1, 1)
            a[i].append(element)
    for i in range(k):
        a[i][i] = -(sum(a[i]) - a[i][i]) + 10 ** (-k)
    return a


def create_f(A, x):
    length = len(x)
    F = [0 for i in range(length)]
    for i in range(length):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            index = A.indices[j]
            a_j = A.data[j]
            F[i] += a_j * x[index]
    return F


def determinant(A):
    L, U = L_U(A)
    det = 1
    for i in U.indptr[:-1]:
        det *= U.data[i]
    return det


def MSE(x_real, x_produced, k):
    return np.linalg.norm(np.array(x_real) - np.array(x_produced)) / k


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
#"Исследование" работы метода при использовании матриц Гильберта
for k in range(3, 20):
    print('Размер матрицы Гильберта: ', k)
    x = [i + 1 for i in range(k)]
    A = create_gilbert_matrix(k)
    rangA = np.linalg.matrix_rank(A)
    A = csr_matrix(A)
    F = np.array(create_f(A, x))
    AF = np.concatenate([A.A, F.reshape(k, 1)], axis=1)
    rangAF = np.linalg.matrix_rank(AF)
    if rangA != rangAF:
        print('Система несовместна!')
        continue
    x1 = gauss_method(A, F)
    print('Матрица A: ', '\n', A.A)
    print('Матрица F: ', '\n', F)
    print('Матрица X: ', '\n', x1)
    print('MSE = ', MSE(x, x1, k))
    print('===========================================================================================')


for k in range(3, 20):
    print('Размер матрицы: ', k)
    x = [i + 1 for i in range(k)]
    A = create_matrix(k)
    rangA = np.linalg.matrix_rank(A)
    A = csr_matrix(A)
    F = np.array(create_f(A, x))
    AF = np.concatenate([A.A, F.reshape(k, 1)], axis=1)
    rangAF = np.linalg.matrix_rank(AF)
    if rangA != rangAF:
        print('Система несовместна!')
        continue
    x1 = gauss_method(A, F)
    print('Матрица A: ', '\n', A.A)
    print('Матрица F: ', '\n', F)
    print('Матрица X: ', '\n', x1)
    print('MSE = ', MSE(x, x1, k))
    print('===========================================================================================')
B = csr_matrix(np.array([[1, 3], [5, 7]]))
L, U = L_U(B)
print(L.A)
print(U.A)
print(gauss_method(B, np.eye(B.shape[0])))
