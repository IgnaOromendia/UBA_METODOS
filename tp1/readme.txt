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

## Creating virtual environment
- To use the library you will have to create a virtual environment
- In order to do this you can simply execute the following commands from the UBA_METODOS folder
    virtualenv venv
    ./\venv\Scripts\activate
- Then from the tp1 folder you will have to install the pip requirements with the following command
    pip install -r requirements.txt
- You are now ready to use the library with all the needed packages already installed in your virtual environment
