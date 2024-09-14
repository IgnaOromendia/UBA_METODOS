import unittest
from desarrollo import *

def matriz_valida_EG_sin_pivoteo(n):
    L = np.random.randint(1,2, size=(n,n)).astype(np.float64)
    U = np.random.randint(1,2, size=(n,n)).astype(np.float64)
    for i in range(n):
        L[i,i] = 1
        for j in range(i+1, n):
            L[i,j] = 0
            U[j,i] = 0
    return L @ U

# Generamos una matriz estrictamente diagonal dominanate, por lo tanto tiene LU
def matriz_edd(n):
    A = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        A[i,i] = 2*n
        for j in range(n):
            if i == j: continue
            A[i,j] = 1
    return A

def matriz_tridiagonal_edd(n):
    A = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        A[i,i] = 3
        if i < n - 1: A[i,i+1] = 1
        if i > 0: A[i,i-1] = 1
    return A

def armar_matriz_tri_con(a,b,c):
    return np.diag(a) + np.diag(b) + np.diag(c)

class TestEliminaciongaussiana(unittest.TestCase):
    def test_01_EG_sin_pivoteo(self):
        for n in range(3, 50):
            A = matriz_valida_EG_sin_pivoteo(n)
            b = np.random.randint(1, 10, size=n).astype(np.float64)

            x = eliminacion_gaussiana(A, b)

            np.testing.assert_array_almost_equal(np.dot(A, x), b, decimal=5)

    def test_02_EG_con_pivoteo(self):
        for n in range(3, 50):
            A = matriz_valida_EG_sin_pivoteo(n)
            b = np.random.randint(1, 10, size=n).astype(np.float64)

            A[0, 0] = 0  # Para verificar que realmente este pivoteando

            x = eliminacion_gaussiana_pivoteo(A.copy(), b)

            np.testing.assert_array_almost_equal(np.dot(A, x), b, decimal=5)

    def test_03_EG_tridiagonal(self):
        for n in range(5, 6):
            d = np.random.randint(1, 10, size=n).astype(np.float64)
            a = np.full(n, 1, dtype=np.float64)
            b = np.full(n, 3, dtype=np.float64)
            c = np.full(n, 1, dtype=np.float64)
            a[0]   = 0
            c[-1]  = 0

            x = eliminacion_gaussiana_tridiagonal(a,b,c,d)
            
            np.testing.assert_array_almost_equal(np.dot(matriz_tridiagonal_edd(n), x), d, decimal=5)

    def test_04_factorizacion_LU(self):
        for n in range(3, 50):
            A = matriz_edd(n) 

            L, U = factorizar_LU(A.copy())

            np.testing.assert_array_almost_equal(np.dot(L, U), A, decimal=5)

            for _ in range(5):
                b = np.random.randint(1, 10, size=n).astype(np.float64)

                x = resolver_LU(L,U,b)

                np.testing.assert_array_almost_equal(np.dot(A, x), b, decimal=5)

    def test_05_factorizacion_LU_tridiagonal(self):
        for n in range(3, 50):
            A = matriz_tridiagonal_edd(n) 

            L, U = factorizar_LU_tri(A.copy())

            np.testing.assert_array_almost_equal(np.dot(L, U), A, decimal=5)

            for _ in range(5):
                b = np.random.randint(1, 10, size=n).astype(np.float64)

                x = resolver_LU(L,U,b)

                np.testing.assert_array_almost_equal(np.dot(A, x), b, decimal=5)

if __name__ == "__main__":
    unittest.main()
