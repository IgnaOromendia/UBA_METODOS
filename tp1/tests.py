import unittest
import numpy as np
import tp1

class TestEliminaciongaussiana(unittest.TestCase):
    def test_01_EG_sin_pivoteo(self):
        for n in range(3,50):
            A = np.random.uniform(1, 10, size=(n, n)).astype(np.float64)
            b = np.random.uniform(1, 10, size=n).astype(np.float64)
            
            x = tp1.eliminacion_gaussiana(A.copy(),b)

            np.testing.assert_array_almost_equal(np.dot(A,x), b, decimal=5)
        
    def test_02_EG_con_pivoteo(self):
        for n in range(3,50):
            A = np.random.uniform(1, 10, size=(n, n)).astype(np.float64)
            b = np.random.uniform(1, 10, size=n).astype(np.float64)

            A[0,0] = 0 # Para verificar que realmente este pivoteando
            
            x = tp1.eliminacion_gaussiana_pivoteo(A.copy(),b)

            np.testing.assert_array_almost_equal(np.dot(A,x), b, decimal=5)

    def test_03_EG_tridiagonal(self):
        for n in range(3,20):
            d_pri = np.random.uniform(1, 10, size=n).astype(np.float64)            
            d_sup = np.random.uniform(1, 10, size=n-1).astype(np.float64)
            d_inf = np.random.uniform(1, 10, size=n-1).astype(np.float64)
            b     = np.random.uniform(1, 10, size=n).astype(np.float64)
            
            A = np.diag(d_pri) + np.diag(d_sup, k=1) + np.diag(d_inf, k=-1)
            
            x = tp1.eliminacion_gaussiana_tridiagonal(A.copy(),b)

            np.testing.assert_array_almost_equal(np.dot(A,x), b, decimal=5)

    def test_04_factorizacion_LU(self):
        for n in range(3,20):
            A = np.random.uniform(1, 10, size=(n, n)).astype(np.float64)

            L,U = tp1.factorizar_LU(A.copy())

            np.testing.assert_array_almost_equal(np.dot(L,U), A, decimal=5)

            for _ in range(5):
                b = np.random.uniform(1, 10, size=n).astype(np.float64)
                
                y = tp1.foward_substitution(L,b)
                x = tp1.backward_substitution(U,y)

                np.testing.assert_array_almost_equal(np.dot(A,x), b, decimal=5) 

    def test_05_factorizacion_LU_tridiagonal(self):
        for n in range(3,20):
            d_pri = np.random.uniform(1, 10, size=n).astype(np.float64)            
            d_sup = np.random.uniform(1, 10, size=n-1).astype(np.float64)
            d_inf = np.random.uniform(1, 10, size=n-1).astype(np.float64)
            
            A = np.diag(d_pri) + np.diag(d_sup, k=1) + np.diag(d_inf, k=-1)

            L,U = tp1.factorizar_LU(A.copy())

            np.testing.assert_array_almost_equal(np.dot(L,U), A, decimal=5)

            for _ in range(5):
                b = np.random.uniform(1, 10, size=n).astype(np.float64)
                
                y = tp1.foward_substitution(L,b)
                x = tp1.backward_substitution(U,y)

                np.testing.assert_array_almost_equal(np.dot(A,x), b, decimal=5)

if __name__ == "__main__":
    unittest.main()