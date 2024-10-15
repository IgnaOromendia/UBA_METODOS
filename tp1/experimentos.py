from desarrollo import *
import sys
import time
from tests import *
import matplotlib.pyplot as plt

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
                l, u1, u2 = factorizar_LU_tri(a, b, c)
                resolver_LU_tri(l, u1, u2, d)
                end_time = time.time()
                tiempo__prec_descartable[i] = end_time - start_time
            else:
                start_time = time.time()
                resolver_LU_tri(l, u1, u2, d)
                end_time = time.time()
                tiempo__prec_descartable[i] = (end_time - start_time) + tiempo__prec_descartable[i - 1]

        if j == 0:
            tiempo__prec = tiempo__prec_descartable
            tiempo_trid = tiempo_trid_descartable
        else:
            if tiempo_trid_descartable[-1] < tiempo_trid[-1]: tiempo_trid = tiempo_trid_descartable
            if tiempo__prec_descartable[-1] < tiempo__prec[-1]: tiempo__prec = tiempo__prec_descartable

    return tiempo_trid, tiempo__prec

def verificar_implementacion_tri(n):
    d1 = generar_d1(n)
    d2 = generar_d2(n)
    d3 = generar_d3(n)

    # Matriz trifiagonal laplaciana
    a, b, c = diagonales(generar_laplaciano(n))

    l, u1, u2 = factorizar_LU_tri(a, b, c)

    w1 = resolver_LU_tri(l, u1, u2, d1)
    w2 = resolver_LU_tri(l, u1, u2, d2)
    w3 = resolver_LU_tri(l, u1, u2, d3)

    tams = [i for i in range(n)]

    plt.plot(tams, w1, label="(a)", color="steelblue")
    plt.plot(tams, w2, label="(b)", color="peru")
    plt.plot(tams, w3, label="(c)", color="forestgreen")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.savefig("graficos/verificacion_triangular.png", format="PNG", bbox_inches='tight')

def plot_diffusion_evolution_2D(alfa=0.1, n=15, m=100, tiempos=None):
    if tiempos is None:
        tiempos = [0, 9, 99]
    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()

    for i, t_i in enumerate(tiempos):
        difusiones = simular_difusion_2D(alfa, n, m)
        color = axs[i].pcolor(difusiones[tiempos[i]], cmap='hot')
        plt.colorbar(color, ax=axs[i], label='u')
        axs[i].set_title(f'Tiempo = ' + str(tiempos[i] + 1))
        axs[i].set_xlabel('k')
        axs[i].set_ylabel('x')
    plt.tight_layout()
    plt.savefig("graficos/mapas_de_calor_2D.png", format="PNG", bbox_inches='tight')

def plot_diffusion_evolution(alfas, n=101, r=10, m=1000):
    fig, axs = plt.subplots(2,2)
    axs = axs.flatten()

    for i,alfa in enumerate(alfas):
        difusiones = simular_difusion(alfa, n, r, m)
        color = axs[i].pcolor(difusiones.T, cmap='hot')
        fig.colorbar(color, ax=axs[i], label='u')
        
        axs[i].set_title('Alfa = ' + str(alfa))
        axs[i].set_xlabel('k')
        axs[i].set_ylabel('x')

    plt.tight_layout()
    plt.savefig("graficos/mapas_de_calor.png", format="PNG", bbox_inches='tight')

def plot_error_numerico():
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
    plt.legend()
    plt.savefig('graficos/grafico_error_numerico.png')
    plt.show()
    
def plot_tridiagonal_vs_precomputo():
    lista_de_size = []
    limite_tam_matrices = 7
    
    for i in range(limite_tam_matrices):
        b = 2 ** i + 1
        lista_de_size.append(b)

    size = 50
    cant_repeticiones = 10
    cant_veces_calcular_tridiagonal = 2 ** 10

    t_tridiagonal2, t_pre_tridiagonal = calcular_tiempos_trid_vs_precomputo_trdi(size, cant_veces_calcular_tridiagonal, cant_repeticiones)

    t_tridiagonal2 = [i * 1000 for i in t_tridiagonal2]
    t_pre_tridiagonal = [i * 1000 for i in t_pre_tridiagonal]

    plt.plot([i for i in range(cant_veces_calcular_tridiagonal)], t_tridiagonal2, color='blue', label='Eliminacion tridiagonal')
    plt.plot([i for i in range(cant_veces_calcular_tridiagonal)], t_pre_tridiagonal, color='red', label='Eliminacion tridiagonal con precomputo')
    
    plt.ylabel('Tiempo en milisegundos')
    plt.xlabel('Cantidad de soluciones buscadas')
    plt.yscale("log",base=10)
    plt.xscale("log", base=2)
    plt.legend()
    plt.savefig('graficos/tridiagonal_vs_precomputo_tridiagonal.png')
    plt.show()

def plot_pivoteo_vs_tridiagonal():
    lista_de_size = []
    limite_tam_matrices = 7
    
    for i in range(limite_tam_matrices):
        b = 2 ** i + 1
        lista_de_size.append(b)
    
    size = 50
    cant_repeticiones = 10
    cant_veces_calcular_tridiagonal = 2 ** 10
    
    t_pivoteo, t_tridiagonal = comparar_tiempo_pivoteo_vs_tridiagonal(lista_de_size, cant_repeticiones)
    

    plt.plot(lista_de_size, t_pivoteo, color='blue', label='Eliminacion con pivoteo')
    plt.plot(lista_de_size, t_tridiagonal, color='red', label='Eliminacion tridiagonal')
    plt.ylabel('Tiempo en segundos')
    plt.xlabel('Tamaño de la matriz')
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.legend()
    plt.savefig('graficos/tridiagonal_vspivoteo.png')
    plt.show()

def comparar_difusion_2d(alfa, m, lista_size):

    lista_tiempos_naive = []
    lista_tiempos_pivoteo = []

    for i in range(len(lista_size)):
        print('En: ', lista_size[i])
        start_time = time.time()
        difusiones = simular_difusion_2D_naive(alfa, lista_size[i], m)
        end_time = time.time()
        lista_tiempos_naive.append(end_time-start_time)

        start_time = time.time()
        difusiones = simular_difusion_2D_con_pivoteo(alfa, lista_size[i], m)
        end_time = time.time()
        lista_tiempos_pivoteo.append(end_time-start_time)
        

    plt.plot(lista_size, lista_tiempos_naive, color='blue', label='EG naive')
    plt.plot(lista_size, lista_tiempos_pivoteo, color='red', label='EG con pivoteo')
    plt.ylabel('Tiempo en milisegundos')
    plt.xlabel('Tamaño de la matriz')
    plt.legend()
    #plt.yscale("log")
    #plt.savefig('graficos/ej72.png')
    plt.show()


if __name__ == "__main__":
    print("Experimentando...")
    # plot_diffusion_evolution(alfas=[0.1,0.3,0.6,1])
    comparar_difusion_2d(0.1, 100, [5,7,10,12,15])
    # plot_diffusion_evolution_2D(0.1, 15, 100, [0,9,49,99])
    # plot_error_numerico()
    # plot_tridiagonal_vs_precomputo()
    # plot_pivoteo_vs_tridiagonal()
    

    

    

    
