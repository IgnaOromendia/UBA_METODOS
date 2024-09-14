import sys
import numpy as np
import matplotlib.pyplot as plt  # type: ignore para que no se queje vs code


# Substitutions

def backward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        suma = 0
        for j in range(i, n):
            suma += A[i, j] * x[j]
        x[i] = (b[i] - suma) / A[i, i]

    return x


def forward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)

    for i in range(n):
        suma = 0
        for j in range(0, i):
            suma += A[i, j] * x[j]
        x[i] = (b[i] - suma) / A[i, i]

    return x


# Eliminacion Gaussiana

def eliminacion_gaussiana(A, b):
    m = 0
    n = A.shape[0]
    A0 = A.copy()
    b0 = b.copy()

    for i in range(0, n):
        if A[i, i] == 0: print("AAAA!")  #TODO
        for j in range(i + 1, n):
            m = A0[j, i] / A0[i, i]
            b0[j] = b0[j] - (m * b0[i])  # aplicar a b
            for k in range(i, n):
                A0[j, k] = A0[j, k] - (m * A0[i, k])

    return backward_substitution(A0, b0)


def permutar_filas(A, i, j):
    copia = A[i].copy()
    A[i] = A[j]
    A[j] = copia


def permutar_elementos_vector(A, i, j):
    copia = A[i]
    A[i] = A[j]
    A[j] = copia


def encontrar_pivote(A, i):
    max = abs(A[i, i])
    fila = i
    # Buscamos la fila con el numero mas grande dentro de la columna i
    for k in range(i, len(A)):
        # Nos quedamos con el numero de la fila para luego permutar
        # Updateamos el nuevo maximo
        if abs(A[k, i]) > max:
            fila = k
            max = abs(A[k, i])
    # Devolvemos la fila
    return fila


def eliminacion_gaussiana_pivoteo(A, b):
    m = 0
    n = A.shape[0]
    A0 = A.copy()
    b0 = b.copy()

    for i in range(0, n):
        fila = encontrar_pivote(A0, i)
        permutar_filas(A0, i, fila)
        permutar_elementos_vector(b0, i, fila)
        for j in range(i + 1, n):
            m = A0[j][i] / A0[i][i]  # coeficiente
            b0[j] = b0[j] - (m * b0[i])  # aplicar a b
            for k in range(i, n):
                A0[j][k] = A0[j][k] - (m * A0[i][k])

    return backward_substitution(A0, b0)


def eliminacion_gaussiana_tridiagonal(T, b):
    n = T.shape[0]
    L = np.eye(n, dtype=np.float64)
    A = np.zeros(n, dtype=np.float64)
    B = np.zeros(n, dtype=np.float64)
    C = np.zeros(n, dtype=np.float64)

    # ai xi−1 + bi xi + ci xi+1 = di

    # Armamos los vectores A B C
    for i in range(n):
        B[i] = T[i][i]
        A[i] = 0 if i == 0 else T[i][i - 1]
        C[i] = 0 if i == n - 1 else T[i][i + 1]

    # Resolvemos
    for i in range(1, n):
        m = A[i] / B[i - 1]
        L[i, i - 1] = m
        A[i] = A[i] - m * B[i - 1]  # A_i - A_i / B_i-1 * B_i-1
        B[i] = B[i] - m * C[i - 1]  # B_i - A_i / B_i-1 * C_i-1

    for i in range(n):
        T[i][i] = B[i]
        if (i >= 1): T[i][i - 1] = A[i]
        if (i < n - 1): T[i][i + 1] = C[i]

    return backward_substitution(T, b)


# Factorización LU


def factorizar_LU(T):
    m = 0
    n = T.shape[0]
    A0 = T.copy()
    L = np.eye(n, dtype=np.float64)

    for i in range(0, n):
        for j in range(i + 1, n):
            m = A0[j, i] / A0[i, i]
            L[j, i] = m
            for k in range(i, n):
                A0[j, k] = A0[j, k] - (m * A0[i, k])

    for i in range(0, n):
        for j in range(0, i):
            A0[i][j] = 0

    return L, A0


def factorizar_LU_tri(T):
    n = T.shape[0]
    L = np.eye(n, dtype=np.float64)
    A = np.zeros(n, dtype=np.float64)
    B = np.zeros(n, dtype=np.float64)
    C = np.zeros(n, dtype=np.float64)

    # ai xi−1 + bi xi + ci xi+1 = di

    # Armamos los vectores A B C
    for i in range(n):
        B[i] = T[i][i]
        A[i] = 0 if i == 0 else T[i][i - 1]
        C[i] = 0 if i == n - 1 else T[i][i + 1]

    # Resolvemos
    for i in range(1, n):
        coeficiente = A[i] / B[i - 1]
        L[i, i - 1] = coeficiente
        A[i] = A[i] - coeficiente * B[i - 1]  # A_i - A_i / B_i-1 * B_i-1
        B[i] = B[i] - coeficiente * C[i - 1]  # B_i - A_i / B_i-1 * C_i-1

    for i in range(n):
        T[i][i] = B[i]
        if (i >= 1): T[i][i - 1] = A[i]
        if (i < n - 1): T[i][i + 1] = C[i]

    return L, T


# Laplaciano

def generar_laplaciano(n):
    T = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(T, -2)
    np.fill_diagonal(T[1:], 1)
    np.fill_diagonal(T[:, 1:], 1)
    return T


def generar_laplaciano_2d(n):
    T = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        T[i, i] = -4
        if i != n - 1 and (i + 1) % 3 != 0:
            T[i, i + 1] = 1
        if i != 0 and i % 3 != 0:
            T[i, i - 1] = 1
        if i <= n - 4:
            T[i, i + 3] = 1
        if i >= 3:
            T[i, i - 3] = 1

    return T


def generar_d1(n):
    d = np.zeros(n, dtype=np.float64)
    d[n // 2 + 1] = 4 / n
    return d


def generar_d2(n):
    return np.full(n, 4 / n ** 2)


def generar_d3(n):
    d = np.zeros(n, dtype=np.float64)
    for i in range(n): d[i] = (-1 + ((2 * i) / (n - 1))) * (12 / (n ** 2))
    return d


def verificar_implementacion_tri(n):
    d1 = generar_d1(n)
    d2 = generar_d2(n)
    d3 = generar_d3(n)

    # Matriz laplaciana
    A = generar_laplaciano(n)

    L, U = factorizar_LU_tri(A.copy())

    y = forward_substitution(L, d1)
    u1 = backward_substitution(U, y)

    y = forward_substitution(L, d2)
    u2 = backward_substitution(U, y)

    y = forward_substitution(L, d3)
    u3 = backward_substitution(U, y)

    tams = [i for i in range(n)]

    plt.plot(tams, u1, label="(a)", color="steelblue")
    plt.plot(tams, u2, label="(b)", color="peru")
    plt.plot(tams, u3, label="(c)", color="forestgreen")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.show()


# Difusión

def generar_u0(n, r):
    u = np.zeros(n, dtype=np.float64)

    lower = (n // 2) - r
    upper = (n // 2) + r

    for i in range(lower + 1, upper): u[i] = 1.0

    return u


def generar_u0_2d(n):
    u = np.zeros((n, n), dtype=np.float64)
    u[(n // 2), (n // 2)] = 100
    return u


def calcular_difusion(A, r, m):
    n = A.shape[0]
    u = [generar_u0(n, r)]

    L, U = factorizar_LU_tri(A.copy())

    for k in range(1, m):
        uk = resolver_LU(L, U, u[k-1])
        u.append(uk)

    return np.array(u)


def resolver_LU(L, U, x):
    y = forward_substitution(L, x)
    uk = backward_substitution(U, y)
    return uk


def calcular_difusion_2d(A, m):
    n = A.shape[0]
    sn = int(np.sqrt(n))
    u = [generar_u0_2d(sn)]
    L, U = factorizar_LU(A.copy())
    for k in range(1, m):
        y = forward_substitution(L, u[k - 1].flatten())
        uk = backward_substitution(U, y)
        # TODO Bordes
        uk = uk.reshape(sn, sn)
        uk[sn // 2, sn // 2] = 100
        uk[:, 0] = 0
        uk[:, sn - 1] = 0
        uk[0, :] = 0
        uk[sn - 1, :] = 0
        u.append(uk)

    return np.array(u)


def simular_difusion_2d(alfa, n, r, m):
    T = generar_laplaciano_2d(n * n)

    A = np.eye(n * n, dtype=np.float64) - alfa * T

    difusiones_2d = calcular_difusion_2d(A, m)

    return difusiones_2d


def plot_diffusion_evolution(alfa=1, n=101, r=10, m=1000):
    difusiones = simular_difusion_2d(alfa, n, r, m)
    print("difusiones", difusiones.shape)
    #plt.pcolor(difusiones[50], cmap='hot')
    plt.colorbar(label='u')
    plt.title(f'Mapa de calor')
    plt.xlabel('k')
    plt.ylabel('x')

if __name__ == "__main__":
    print("TP1")
    #simular_difusion_2d(0.1, 15, 1, 100)
    plot_diffusion_evolution(1, 3, 1, 100)
