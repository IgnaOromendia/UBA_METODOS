from desarrollo import *

def explorar_error_numerico(lista_epsilons, fl):
    lista_resultados = []
    
    
    for i in range(len(lista_epsilons)):
        eps = lista_epsilons[i]
        A = np.array([[1, 2+eps, 3-eps],
                [1-eps, 2, 3+eps],
                [1+eps, 2-eps, 3]], dtype=fl)
        x = np.array([1,1,1], dtype=fl)
        b = np.array([6,6,6], dtype=fl)
        v = eliminacion_gaussiana_pivoteo(A, b)
        lista_resultados.append(np.max(abs(x-v)))

    return lista_resultados

def hacer_lista_eps():
        eps = 1
        for i in range(10**5):
             eps = eps - 10**-6
        print(eps)



if __name__ == "__main__":
    #hacer_lista_eps()
    #exponents = np.linspace(-6, 0, num=10, endpoint=False)
    #powers_of_ten = 10 ** exponents

    log_space_values = np.logspace(np.log10(10**-6), np.log10(1), num=100)
    lista_epsilons =log_space_values
    print(log_space_values)
    #lista_epsilons = np.arange(10**-6, 1, 10**-1)
    a = explorar_error_numerico(lista_epsilons, np.float64)
    b = explorar_error_numerico(lista_epsilons, np.float32)

    
    plt.plot(lista_epsilons, a, 'o',color = 'blue', label="64bits")
    plt.plot(lista_epsilons, b,'o',color = 'red', label="32bits")
    
    #plt.ylabel('Tiempo en segundos')
    #plt.xlabel('Cant de iteraciones')
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
