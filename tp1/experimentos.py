from desarrollo import *
import sys
import time
from tests import *


def comparar_tiempo_pivoteo_vs_tridiagonal(lista_size, cant_repeticiones):
    tiempo_pivoteo = []
    tiempo_tridiagonal = []

    for i in range(len(lista_size)):
        A = generar_laplaciano(lista_size[i])
        np.random.seed(1)
        d = np.random.randint(1, 10, size=len(A))
        a, b, c = diagonales(A)
        k = []
        h = []

        for j in range(cant_repeticiones):
            start_time = time.time()
            eliminacion_gaussiana_pivoteo(A, d)

            end_time = time.time()
            k.append(end_time - start_time)
        tiempo_pivoteo.append(min(k))

        for j in range(cant_repeticiones):
            start_time = time.time()
            eliminacion_gaussiana_tridiagonal(a, b, c, d)

            end_time = time.time()
            h.append(end_time - start_time)
        tiempo_tridiagonal.append(min(h))
    return tiempo_pivoteo, tiempo_tridiagonal


def explorar_error_numerico(lista_epsilons, fl):
    lista_resultados = []

    for i in range(len(lista_epsilons)):
        eps = lista_epsilons[i]
        A = np.array([[1, 2 + eps, 3 - eps],
                      [1 - eps, 2, 3 + eps],
                      [1 + eps, 2 - eps, 3]], dtype=fl)
        x = np.array([1, 1, 1], dtype=fl)
        b = np.array([6, 6, 6], dtype=fl)
        v = eliminacion_gaussiana_pivoteo(A, b)
        lista_resultados.append(np.max(abs(x - v)))

    return lista_resultados


def calcular_tiempos_trid_vs_precomputo_trdi(size, cant_repeticiones, n):
    tiempo_trid = []
    tiempo__prec = []
    A = generar_laplaciano(size)

    a, b, c = diagonales(A)

    for j in range(n):
        tiempo_trid_descartable = [0 for i in range(cant_repeticiones)]
        tiempo__prec_descartable = [0 for i in range(cant_repeticiones)]
        np.random.seed(9)

        for i in range(cant_repeticiones):
            d = np.random.randint(1, 10, size)
            start_time = time.time()
            eliminacion_gaussiana_tridiagonal(a, b, c, d)
            end_time = time.time()
            if i == 0:
                tiempo_trid_descartable[i] = end_time - start_time
            else:
                tiempo_trid_descartable[i] = (end_time - start_time) + tiempo_trid_descartable[i - 1]

        np.random.seed(9)
        for i in range(cant_repeticiones):
            d = np.random.randint(1, 10, size)
            if i == 0:
                start_time = time.time()
                L, U = factorizar_LU_tri(a, b, c)
                y = forward_substitution_tri(L, d)
                x = backward_substitution_tri(U, y)
                end_time = time.time()
                tiempo__prec_descartable[i] = end_time - start_time
            else:
                start_time = time.time()
                y = forward_substitution_tri(L, d)
                x = backward_substitution_tri(U, y)
                end_time = time.time()
                tiempo__prec_descartable[i] = (end_time - start_time) + tiempo__prec_descartable[i - 1]

        if j == 0:
            tiempo__prec = tiempo__prec_descartable
            tiempo_trid = tiempo_trid_descartable
        else:
            if tiempo_trid_descartable[-1] < tiempo_trid[-1]: tiempo_trid = tiempo_trid_descartable
            if tiempo__prec_descartable[-1] < tiempo__prec[-1]: tiempo__prec = tiempo__prec_descartable

    return tiempo_trid, tiempo__prec


if __name__ == "__main__":
    # lista_de_size = []
    # limite_tam_matrices = 7
    #
    # for i in range(limite_tam_matrices):
    #     b = 2 ** i + 1
    #     lista_de_size.append(b)
    #
    # size = 50
    # cant_repeticiones = 10
    # cant_veces_calcular_tridiagonal = 2 ** 10
    #
    # t_pivoteo, t_tridiagonal = comparar_tiempo_pivoteo_vs_tridiagonal(lista_de_size, cant_repeticiones)
    # t_tridiagonal2, t_pre_tridiagonal = calcular_tiempos_trid_vs_precomputo_trdi(size, cant_veces_calcular_tridiagonal,
    #                                                                              cant_repeticiones)
    # TODO Comparar todos los algoritmos de eliminacion gaussian con matrices arbitrarias
    # - Naive sin pivoteo
    # - Con Pivoteo
    # - Factorizacion LU
    # Creo las matrices usando factorizaciones LU para garantizar
    n = list(np.power(2, np.arange(9)))
    tiempo_naive = []
    tiempo_pivoteo = []
    tiempo_LU = []
    for i in range(len(n)):
        A = matriz_valida_EG_sin_pivoteo(n[i])
        b = np.random.randint(1, 10, n[i])

        start_time = time.time()
        eliminacion_gaussiana(A, b)
        end_time = time.time()
        tiempo_naive.append(end_time - start_time)

        start_time = time.time()
        eliminacion_gaussiana_pivoteo(A, b)
        end_time = time.time()
        tiempo_pivoteo.append(end_time - start_time)

        start_time = time.time()
        L, U = factorizar_LU(A)
        resolver_LU(L, U, b)
        end_time = time.time()
        tiempo_LU.append(end_time - start_time)

    # Printeo los tiempos
    for i in range(len(n)):
        print(tiempo_naive[i])
        print(tiempo_pivoteo[i])
        print(tiempo_LU[i])
    plt.plot(n, tiempo_naive, color='blue', label='Eliminacion Gaussiana Naive')
    plt.plot(n, tiempo_pivoteo, color='red', label='Eliminacion Pivoteo')
    plt.plot(n, tiempo_LU, color='green', label='Eliminacion con LU')
    plt.ylabel('Tiempo en segundos')
    plt.xlabel('Tamaño de la matriz')
    plt.xscale("log", base=2)
    plt.yscale("log", base=10)
    plt.legend()
    plt.show()
    sys.exit()

    plt.plot(lista_de_size, t_pivoteo, color='blue', label='Eliminacion con pivoteo')
    plt.plot(lista_de_size, t_tridiagonal, color='red', label='Eliminacion tridiagonal')
    plt.ylabel('Tiempo en segundos')
    plt.xlabel('Tamaño de la matriz')
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.legend()
    plt.savefig('graficos/tridiagonal_vspivoteo.png')
    plt.show()

    plt.plot([i for i in range(cant_veces_calcular_tridiagonal)], t_tridiagonal2, color='blue',
             label='Eliminacion tridiagonal')
    plt.plot([i for i in range(cant_veces_calcular_tridiagonal)], t_pre_tridiagonal, color='red',
             label='Eliminacion tridiagonal con precomputo')
    plt.ylabel('Tiempo en segundos')
    plt.xlabel('Cantidad de soluciones buscadas')
    plt.xscale("log", base=2)
    plt.legend()
    plt.savefig('graficos/tridiagonal_vs_precomputo_tridiagonal.png')
    plt.show()

    lista_epsilons = np.logspace(np.log10(10 ** -6), np.log10(1), num=100)

    a = explorar_error_numerico(lista_epsilons, np.float64)
    b = explorar_error_numerico(lista_epsilons, np.float32)

    plt.plot(lista_epsilons, a, 'o', color='blue', label="64bits")
    plt.plot(lista_epsilons, b, 'o', color='red', label="32bits")

    plt.ylabel('Error numerico')
    plt.xlabel('Epsilon')
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Error numerico en funcion del epsilon")

    plt.savefig('graficos/grafico_error_numerico.png')
    plt.legend()
    plt.show()
