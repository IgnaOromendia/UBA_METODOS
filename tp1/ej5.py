from desarrollo import *
import time

def calcular_tiempo_avr(funcion_hacer_matriz,funcion_calcular_sistema, l, h, s, cant):
    time_avg = 0
    for i in range(cant):
        start_time = time.time()
        funcion_hacer_matriz(l,h,s,funcion_calcular_sistema)
        end_time = time.time() 
        time_avg = time_avg + (end_time-start_time)
    
    return (time_avg/cant)

def calcular_sistema_de_n_elementos(l, h, s, f):
  np.random.seed(9)
  A = np.random.uniform(l, h, (s,s))
  b = np.random.uniform(l, h, s)
  return f(A, b)

def calcular_sistema_de_n_elementos_tridiagonal(l, h, s, f):
    np.random.seed(9)
    a = np.random.uniform(l, h, s-1)
    b = np.random.uniform(l, h, s)
    c = np.random.uniform(l, h, s-1)
    T = np.zeros((s,s), dtype=np.float64)
    for i in range(s):
        T[i,i] = b[i]
    for i in range(s-1):
        T[i, i+1] = c[i]
        T[i+1, i] = a[i]    
    return f(T, b)

def calcular_tiempos_naive_vs_pivoteo(l):
    tiempos_pivoteo = []
    tiempos_naive = []
    i = 0
    for s in l:
        t2 =  calcular_tiempo_avr(calcular_sistema_de_n_elementos,eliminacion_gaussiana_pivoteo, 10**-3,10**3, s, 10)
        t1 =  calcular_tiempo_avr(calcular_sistema_de_n_elementos,eliminacion_gaussiana, 10**-3,10**3, s, 10)
        tiempos_naive.append(t1)
        tiempos_pivoteo.append(t2)
        #print("Tiempo nainve: ", t1, " Tiempo pivoteo: ", t2, "para el tamaño: ", s)
        i = i+1 
    
    return tiempos_naive, tiempos_pivoteo

def calcular_tiempos_naive_vs_tridiagonal(l):
    tiempos_naive = []
    tiempos_trid = []
    i = 0
    for s in l:
        t1 =  calcular_tiempo_avr(calcular_sistema_de_n_elementos_tridiagonal,eliminacion_gaussiana, 10**-3,10**3, s, 10)
        t2 =  calcular_tiempo_avr(calcular_sistema_de_n_elementos_tridiagonal,eliminacion_gaussiana_tridiagonal, 10**-3,10**3, s, 10)
        tiempos_naive.append(t1)
        tiempos_trid.append(t2)
        #print("Tiempo nainve: ", t1, " Tiempo pivoteo: ", t2, "para el tamaño: ", s)
        i = i+1 
    
    return tiempos_naive, tiempos_trid

def calcular_tiempos_trid_vs_precomputo_trdi(s, n):
    tiempo_trid = [0 for i in range(n)]
    tiempo__prec = [0 for i in range(n)]
    np.random.seed(9)
    A = np.random.uniform(10**-3,10**3, (s,s))
    b = np.random.uniform(10**-3,10**3, s)
    for i in range(n):
        start_time = time.time()
        eliminacion_gaussiana_tridiagonal(A, b)
        end_time = time.time()
        if i == 0:
            tiempo_trid[i] = end_time-start_time
        else: tiempo_trid[i] = (end_time-start_time) + tiempo_trid[i-1]
    for i in range(n):
        if i == 0:
            start_time = time.time()
            L,U = factorizar_LU_tri(A)
            x = backward_substitution(U,y)
            end_time = time.time()
            tiempo__prec[i] =  end_time- start_time
        else:
            start_time = time.time()
            x = backward_substitution(U,y)
            end_time = time.time()
            tiempo__prec[i] = (end_time-start_time) + tiempo__prec[i-1]
    return tiempo_trid, tiempo__prec
        

if __name__ == "__main__":
    a = []
    b = 20
    for i in range(30):
        a.append(b)
        b = b+20

    lista_size = a
    s = 300
    n = 500
    calcular_tiempos_trid_vs_precomputo_trdi(10, 10)
    calcular_tiempos_naive_vs_tridiagonal([10])
    calcular_tiempos_naive_vs_pivoteo([10])

    t_t,t_pre_t = calcular_tiempos_trid_vs_precomputo_trdi(s, n)
    t_n2, t_t2 = calcular_tiempos_naive_vs_tridiagonal(a)
    t_n3, t_p3 = calcular_tiempos_naive_vs_pivoteo(lista_size)


    #print(t_t)
    #print(t_pre_t)
    #t_n, t_p = calcular_tiempos_naive_vs_pivoteo(lista_size)
    #t_n, t_t = calcular_tiempos_naive_vs_tridiagonal(a)
    plt.plot([i for i in range(n)], t_t, color = 'blue',label='Eliminacion tridiagonal')
    plt.plot([i for i in range(n)], t_pre_t, color = 'red',label='Eliminacion tridiagonal con precomputo')
    plt.ylabel('Tiempo en segundos')
    plt.xlabel('Tamaño de la matriz')
    #plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    plt.show()

    plt.plot(lista_size, t_n2, color = 'blue',label='Eliminacion naive')
    plt.plot(lista_size, t_t2, color = 'red',label='Eliminacion tridiagonal ')
    plt.ylabel('Tiempo en segundos')
    plt.xlabel('Tamaño de la matriz')
    #plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    plt.show()

    plt.plot(lista_size, t_n3, color = 'blue',label='Eliminacion naive')
    plt.plot(lista_size, t_p3, color = 'red',label='Eliminacion pivoteo ')
    plt.ylabel('Tiempo en segundos')
    plt.xlabel('Tamaño de la matriz')
    #plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    plt.show()