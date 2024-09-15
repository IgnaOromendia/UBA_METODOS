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
        

def calcular_tiempos_trid_vs_precomputo_trdi(size, cant_repeticiones):
    tiempo_trid = [0 for i in range(cant_repeticiones)]
    tiempo__prec = [0 for i in range(cant_repeticiones)]
    np.random.seed(9)
    A = generar_laplaciano(size) 
    d = np.random.randint(1,10, size)
    a,b,c = diagonales(A) 

    
    for i in range(cant_repeticiones):
        start_time = time.time()
        eliminacion_gaussiana_tridiagonal(a,b,c,d)
        end_time = time.time()
        if i == 0:
            tiempo_trid[i] = end_time-start_time
        else: tiempo_trid[i] = (end_time-start_time) + tiempo_trid[i-1]

    for i in range(cant_repeticiones):
        if i == 0:
            start_time = time.time()
            L,U = factorizar_LU_tri(a,b,c)
            y = forward_substitution_tri(L, d)
            x = backward_substitution_tri(U,y)
            end_time = time.time()
            tiempo__prec[i] =  end_time- start_time
        else:
            start_time = time.time()
            y = forward_substitution_tri(L, d)
            x = backward_substitution_tri(U,y)
            end_time = time.time()
            tiempo__prec[i] = (end_time-start_time) + tiempo__prec[i-1]
    return tiempo_trid, tiempo__prec


if __name__ == "__main__":
    lista_de_size = []
    b = 2
    for i in range(8):
        i+=1
        b = 2**i
        lista_de_size.append(b)

    size = 500
    cant_repeticiones = 5

    t_pivoteo, t_tridiagonal = comparar_tiempo_pivoteo_vs_tridiagonal(lista_de_size, cant_repeticiones)
    t_tridiagonal2, t_pre_tridiagonal = calcular_tiempos_trid_vs_precomputo_trdi(size, cant_repeticiones)

    plt.plot(lista_de_size, t_pivoteo, color = 'blue',label='Eliminacion con pivoteo')
    plt.plot(lista_de_size, t_tridiagonal, color = 'red',label='Eliminacion tridiagonal')
    plt.ylabel('Tiempo en segundos')
    plt.xlabel('Cantidad de soluciones buscadas')
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    #plt.savefig('graficos/tridiagonal_vspivoteo.png')
    plt.show()

    plt.plot([i for i in range(cant_repeticiones)], t_tridiagonal2, color = 'blue',label='Eliminacion tridiagonal')
    plt.plot([i for i in range(cant_repeticiones)], t_pre_tridiagonal, color = 'red',label='Eliminacion tridiagonal con precomputo')
    plt.ylabel('Tiempo en segundos')
    plt.xlabel('Cantidad de soluciones buscadas')
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig('graficos/tridiagonal_vs_precomputo_tridiagonal.png')
    plt.show()











# def calcular_tiempo_avr(funcion_hacer_matriz,funcion_calcular_sistema, l, h, s, cant):
#     time_avg = 0
#     for i in range(cant):
#         start_time = time.time()
#         funcion_hacer_matriz(l,h,s,funcion_calcular_sistema)
#         end_time = time.time() 
#         time_avg = time_avg + (end_time-start_time)
    
#     return (time_avg/cant)

# def calcular_tiempo_min(funcion_hacer_matriz,funcion_calcular_sistema, l, h, s, cant):
#     time_list = []
#     for i in range(cant):
#         start_time = time.time()
#         funcion_hacer_matriz(l,h,s,funcion_calcular_sistema)
#         end_time = time.time() 
#         time_list.append(end_time-start_time)
    
#     return min(time_list)



# def calcular_sistema_de_n_elementos(l, h, s, f):
#   np.random.seed(9)
#   A = matriz_valida_EG_sin_pivoteo(s)
#   b = np.random.randint(l, h, s)
#   return f(A, b)

# def calcular_sistema_de_n_elementos_tridiagonal_para_matriz(l, h, s, f):
#     np.random.seed(9)
#     T = matriz_tridiagonal_edd(s)
#     b = np.random.uniform(l, h, s)
    
#     return f(T, b)

# def calcular_sistema_de_n_elementos_tridiagonal_para_vectores(l, h, s, f):
#     np.random.seed(9)
#     T = matriz_tridiagonal_edd(s)

#     a, b, c = diagonales(T)


#     d = np.random.uniform(l, h, s)
    
#     return f(a,b, c, d)


# def calcular_tiempos_naive_vs_pivoteo(l):
#     tiempos_pivoteo = []
#     tiempos_naive = []
#     i = 0
#     for s in l:
#         t2 =  calcular_tiempo_min(calcular_sistema_de_n_elementos,eliminacion_gaussiana_pivoteo, 1,20, s, 7)
#         t1 =  calcular_tiempo_min(calcular_sistema_de_n_elementos,eliminacion_gaussiana, 1,20, s, 7)
#         tiempos_naive.append(t1)
#         tiempos_pivoteo.append(t2)
#         #print("Tiempo nainve: ", t1, " Tiempo pivoteo: ", t2, "para el tamaño: ", s)
#         i = i+1 
    
#     return tiempos_naive, tiempos_pivoteo

# def calcular_tiempos_naive_vs_tridiagonal(l):
#     tiempos_naive = []
#     tiempos_trid = []
#     i = 0
#     for s in l:
#         t1 = calcular_tiempo_min(calcular_sistema_de_n_elementos_tridiagonal_para_matriz,eliminacion_gaussiana, 1, 20, s, 7)
#         t2 = calcular_tiempo_min(calcular_sistema_de_n_elementos_tridiagonal_para_vectores,eliminacion_gaussiana_tridiagonal, 10**-3,10**3, s, 7) #TODO CAMBIAR EL 3
        
#         tiempos_naive.append(t1)
#         tiempos_trid.append(t2)
#         #print("Tiempo nainve: ", t1, " Tiempo pivoteo: ", t2, "para el tamaño: ", s)
#         i = i+1 
    
#     return tiempos_naive, tiempos_trid

# def calcular_tiempos_trid_vs_precomputo_trdi(s, n):
#     tiempo_trid = [0 for i in range(n)]
#     tiempo__prec = [0 for i in range(n)]
#     np.random.seed(9)
#     A = matriz_tridiagonal_edd(s) 
#     d = np.random.randint(1,20, s)
#     a,b,c = diagonales(A) 

    
#     for i in range(n):
#         start_time = time.time()
#         eliminacion_gaussiana_tridiagonal(a,b,c,d)
#         end_time = time.time()
#         if i == 0:
#             tiempo_trid[i] = end_time-start_time
#         else: tiempo_trid[i] = (end_time-start_time) + tiempo_trid[i-1]

#     for i in range(n):
#         if i == 0:
#             start_time = time.time()
#             L,U = factorizar_LU_tri(a,b,c)
#             y = forward_substitution_tri(L, d)
#             x = backward_substitution_tri(U,y)
#             end_time = time.time()
#             tiempo__prec[i] =  end_time- start_time
#         else:
#             start_time = time.time()
#             y = forward_substitution_tri(L, d)
#             x = backward_substitution_tri(U,y)
#             end_time = time.time()
#             tiempo__prec[i] = (end_time-start_time) + tiempo__prec[i-1]
#     return tiempo_trid, tiempo__prec
        

# if __name__ == "__main__":
#     a = []
#     b = 2
#     for i in range(9):
#         i+=1
#         b = 2**i
#         a.append(b)

#     lista_size = a
#     s = 300
#     n = 500

#     t_t,t_pre_t = calcular_tiempos_trid_vs_precomputo_trdi(s, n)
#     t_n2, t_t2 = calcular_tiempos_naive_vs_tridiagonal(a)
#     t_n3, t_p3 = calcular_tiempos_naive_vs_pivoteo(lista_size)


    
#     plt.plot([i for i in range(n)], t_t, color = 'blue',label='Eliminacion tridiagonal')
#     plt.plot([i for i in range(n)], t_pre_t, color = 'red',label='Eliminacion tridiagonal con precomputo')
#     plt.ylabel('Tiempo en segundos')
#     plt.xlabel('Cantidad de soluciones buscadas')
#     #plt.xscale("log")
#     #plt.yscale("log")
#     plt.legend()
#     plt.savefig('graficos/tridiagonal_vs_precomputo_tridiagonal.png')
#     plt.show()

#     plt.plot(lista_size, t_n2, color = 'blue',label='Eliminacion naive')
#     plt.plot(lista_size, t_t2, color = 'red',label='Eliminacion tridiagonal ')
#     plt.ylabel('Tiempo en segundos')
#     plt.xlabel('Tamaño de la matriz')
#     plt.xscale("log", base=2)
#     #plt.yscale("log")
#     plt.legend()
#     plt.savefig('graficos/naive_vs_tridiagonal.png')
#     plt.show()

#     plt.plot(lista_size, t_n3, color = 'blue',label='Eliminacion naive')
#     plt.plot(lista_size, t_p3, color = 'red',label='Eliminacion pivoteo ')
#     plt.ylabel('Tiempo en segundos')
#     plt.xlabel('Tamaño de la matriz')
#     plt.xscale("log", base=2)
#     #plt.yscale("log")
#     plt.legend()
#     plt.savefig('graficos/naive_vs_pivoteo.png')
#     plt.show()