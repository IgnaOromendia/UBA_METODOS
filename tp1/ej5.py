from desarrollo import *
import time
from tests import *

def comparar_tiempo_pivoteo_vs_tridiagonal(lista_size, cant_repeticiones):
    tiempo_pivoteo = []
    tiempo_tridiagonal = []


    for i in range(len(lista_size)):
        A = generar_laplaciano(lista_size[i])
        np.random.seed(1)
        d = np.random.randint(1,10,size=len(A))
        a,b,c = diagonales(A)
        k = []
        h = []

        for j in range(cant_repeticiones):
            start_time = time.time()
            eliminacion_gaussiana_pivoteo(A, d)
            
            end_time = time.time() 
            k.append(end_time-start_time)
        tiempo_pivoteo.append(min(k))
        
        for j in range(cant_repeticiones):
            start_time = time.time()
            eliminacion_gaussiana_tridiagonal(a,b,c,d)

            end_time = time.time() 
            h.append(end_time-start_time)
        tiempo_tridiagonal.append(min(h))
    return tiempo_pivoteo, tiempo_tridiagonal
        

def calcular_tiempos_trid_vs_precomputo_trdi(size, cant_repeticiones, n):
    tiempo_trid = []
    tiempo__prec = []
    A = generar_laplaciano(size) 
    
    a,b,c = diagonales(A) 

    for j in range(n):
        tiempo_trid_descartable = [0 for i in range(cant_repeticiones)]
        tiempo__prec_descartable = [0 for i in range(cant_repeticiones)]
        np.random.seed(9)


        for i in range(cant_repeticiones):
            d = np.random.randint(1,10, size)
            start_time = time.time()
            eliminacion_gaussiana_tridiagonal(a,b,c,d)
            end_time = time.time()
            if i == 0:
                tiempo_trid_descartable[i] = end_time-start_time
            else: tiempo_trid_descartable[i] = (end_time-start_time) + tiempo_trid_descartable[i-1]

        np.random.seed(9)
        for i in range(cant_repeticiones):
            d = np.random.randint(1,10, size)
            if i == 0:
                start_time = time.time()
                L,U = factorizar_LU_tri(a,b,c)
                y = forward_substitution_tri(L, d)
                x = backward_substitution_tri(U,y)
                end_time = time.time()
                tiempo__prec_descartable[i] =  end_time- start_time
            else:
                start_time = time.time()
                y = forward_substitution_tri(L, d)
                x = backward_substitution_tri(U,y)
                end_time = time.time()
                tiempo__prec_descartable[i] = (end_time-start_time) + tiempo__prec_descartable[i-1]

        if j == 0:
            tiempo__prec = tiempo__prec_descartable
            tiempo_trid = tiempo_trid_descartable
        else:
            if tiempo_trid_descartable[-1] < tiempo_trid[-1]:tiempo_trid= tiempo_trid_descartable 
            if tiempo__prec_descartable[-1] < tiempo__prec[-1]:tiempo__prec = tiempo__prec_descartable

    return tiempo_trid, tiempo__prec


if __name__ == "__main__":
    lista_de_size = []

    j = 1
    for i in range(7):
        b = 2**j
        lista_de_size.append(b)
        j += 1
        

    size = 50
    cant_repeticiones = 10
    cant_veces_calcular_tridiagonal = 2**10

    t_pivoteo, t_tridiagonal = comparar_tiempo_pivoteo_vs_tridiagonal(lista_de_size, cant_repeticiones)
    t_tridiagonal2, t_pre_tridiagonal = calcular_tiempos_trid_vs_precomputo_trdi(size, cant_veces_calcular_tridiagonal, cant_repeticiones)

    plt.plot(lista_de_size, t_pivoteo, color = 'blue',label='Eliminacion con pivoteo')
    plt.plot(lista_de_size, t_tridiagonal, color = 'red',label='Eliminacion tridiagonal')
    plt.ylabel('Tiempo en segundos')
    plt.xlabel('Tamaño de la matriz')
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.legend()
    #plt.savefig('graficos/tridiagonal_vspivoteo.png')
    plt.show()

    plt.plot([i for i in range(cant_veces_calcular_tridiagonal)], t_tridiagonal2, color = 'blue',label='Eliminacion tridiagonal')
    plt.plot([i for i in range(cant_veces_calcular_tridiagonal)], t_pre_tridiagonal, color = 'red',label='Eliminacion tridiagonal con precomputo')
    plt.ylabel('Tiempo en segundos')
    plt.xlabel('Cantidad de soluciones buscadas')
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.legend()
    plt.savefig('graficos/tridiagonal_vs_precomputo_tridiagonal.png')
    plt.show()

