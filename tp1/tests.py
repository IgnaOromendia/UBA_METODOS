import unittest
from desarrollo import *


class TestEliminaciongaussiana(unittest.TestCase):
    def test_01_EG_sin_pivoteo(self):
        for n in range(3, 50):
            A = np.random.uniform(1, 10, size=(n, n)).astype(np.float64)
            b = np.random.uniform(1, 10, size=n).astype(np.float64)

            x = eliminacion_gaussiana(A.copy(), b)

            np.testing.assert_array_almost_equal(np.dot(A, x), b, decimal=5)

    def test_02_EG_con_pivoteo(self):
        for n in range(3, 50):
            A = np.random.uniform(1, 10, size=(n, n)).astype(np.float64)
            b = np.random.uniform(1, 10, size=n).astype(np.float64)

            A[0, 0] = 0  # Para verificar que realmente este pivoteando

            x = eliminacion_gaussiana_pivoteo(A.copy(), b)

            np.testing.assert_array_almost_equal(np.dot(A, x), b, decimal=5)

    def test_03_EG_tridiagonal(self):
        for n in range(3, 20):
            d_pri = np.random.uniform(1, 10, size=n).astype(np.float64)
            d_sup = np.random.uniform(1, 10, size=n - 1).astype(np.float64)
            d_inf = np.random.uniform(1, 10, size=n - 1).astype(np.float64)
            b = np.random.uniform(1, 10, size=n).astype(np.float64)

            A = np.diag(d_pri) + np.diag(d_sup, k=1) + np.diag(d_inf, k=-1)

            x = eliminacion_gaussiana_tridiagonal(A.copy(), b)

            np.testing.assert_array_almost_equal(np.dot(A, x), b, decimal=5)

    def test_04_factorizacion_LU(self):
        for n in range(3, 20):
            A = np.random.uniform(1, 10, size=(n, n)).astype(np.float64)

            L, U = factorizar_LU(A.copy())

            np.testing.assert_array_almost_equal(np.dot(L, U), A, decimal=5)

            for _ in range(5):
                b = np.random.uniform(1, 10, size=n).astype(np.float64)

                y = forward_substitution(L, b)
                x = backward_substitution(U, y)

                np.testing.assert_array_almost_equal(np.dot(A, x), b, decimal=5)

    def test_05_factorizacion_LU_tridiagonal(self):
        for n in range(3, 20):
            d_pri = np.random.uniform(1, 10, size=n).astype(np.float64)
            d_sup = np.random.uniform(1, 10, size=n - 1).astype(np.float64)
            d_inf = np.random.uniform(1, 10, size=n - 1).astype(np.float64)

            A = np.diag(d_pri) + np.diag(d_sup, k=1) + np.diag(d_inf, k=-1)

            L, U = factorizar_LU(A.copy())

            np.testing.assert_array_almost_equal(np.dot(L, U), A, decimal=5)

            for _ in range(5):
                b = np.random.uniform(1, 10, size=n).astype(np.float64)

                y = forward_substitution(L, b)
                x = backward_substitution(U, y)

                np.testing.assert_array_almost_equal(np.dot(A, x), b, decimal=5)


if __name__ == "__main__":
    unittest.main()


def es_tridiagonal(A):
    rows, cols = A.shape

    if rows != cols:
        return False  # Not a square matrix

    # Iterate over all elements to check if it's a tridiagonal matrix
    for i in range(rows):
        for j in range(cols):
            if abs(i - j) > 1 and A[i, j] != 0:
                return False

    return True


