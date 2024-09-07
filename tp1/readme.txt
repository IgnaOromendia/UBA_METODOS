# Gaussian Elimination Algorithms

This repository contains a Python implementation of various Gaussian elimination algorithms, including naive elimination, elimination with pivoting, LU factorization, and solving tridiagonal systems. The algorithms use `NumPy` for matrix and vector operations.

## Features

1. **Matrix-Vector Multiplication**:
   - Multiplies a given matrix by a vector using NumPy's `dot` function.

2. **Naive Gaussian Elimination**:
   - Solves a system of linear equations using naive Gaussian elimination without pivoting.

3. **Gaussian Elimination with Pivoting**:
   - Solves a system of linear equations using Gaussian elimination with partial pivoting for better numerical stability.

4. **LU Factorization**:
   - Performs LU decomposition on a given matrix and solves the system of equations.

5. **Solving Tridiagonal Systems**:
   - Efficiently solves tridiagonal linear systems using Gaussian elimination.

## Usage

- To use the library, simply import the functions and pass the appropriate matrices and vectors.
- Example usage of `eliminacion_gausseana_naive`:
  ```python
  A = np.array([[1, 2, 1], [1, 0, 1], [0, 1, 2]], dtype=np.float64)
  b = np.array([0, 2, 1], dtype=np.float64)
  x = eliminacion_gausseana_naive(A, b)