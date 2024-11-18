import matplotlib.pyplot as plt
import multiprocessing as mp
import desarrollo as ds

def plot_error_sujeto_1():
    grados, err_ajuste, err_val = ds.predecir_sin_reg(sujeto=1)

    plt.plot(grados, err_ajuste,'.-', label='Ajuste')
    plt.plot(grados, err_val,'.-', label='Val')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Grado')
    plt.ylabel('Error')
    plt.title('ECM de ajuste y validación')
    plt.savefig('graficos/grafico_error_sujeto1.png')
    plt.show()

def heatmap_sujetos(error, sujeto, ax):
    cax = ax.pcolor(error, cmap='hot', shading='auto') 
    plt.colorbar(cax, ax=ax, label='Error')
    
    ax.set_title(f'Mapa de Calor del error del sujeto nº {sujeto}')
    ax.set_ylabel('Grado')
    ax.set_xlabel('$\lambda$')

def explorar_en_sujeto(sujeto):
    x_aju, y_aju = ds.leer_datos('./datos/ajuste.txt', sujeto)
    x_val, y_val = ds.leer_datos('./datos/validacion.txt', sujeto)

    max_g = 2*x_aju.shape[0]
    max_l = 50

    lambdas = [i for i in range(1, max_l)]
    grados  = [i for i in range(1, max_g)]

    g_opt = -1
    l_opt = -1
    error = [[0] * max_l for _ in range(max_g)]
    min_err = 1e8

    for g in grados:
        for l in lambdas:
            error[g][l] = ds.predecir_con_reg(x_aju, y_aju, x_val, y_val, g, l)
            if error[g][l] < min_err:
                min_err = error[g][l]
                g_opt = g
                l_opt = l

    print("Sujeto nº" + str(sujeto) + " terminó")

    return sujeto, error, (g_opt, l_opt)

def explorar_hiperparametros():
    sujetos = [[s] for s in range(1,6)]

    with mp.Pool(processes=5) as pool:
        resultados = pool.starmap(explorar_en_sujeto, sujetos)

    optimos = [None] * 5
    errores = [None] * 5
    
    for sujeto, error, opt in resultados:
        optimos[sujeto-1] = opt
        errores[sujeto-1] = error

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    axes = axes.flatten()

    axes[-1].axis('off')

    for sujeto, error, _ in resultados:
        heatmap_sujetos(error, sujeto, axes[sujeto-1])

    plt.tight_layout()
    plt.savefig("graficos/heatmap.png")
    plt.show()


if __name__ == "__main__":
    print("Experimentando...")
    # plot_error_sujeto_1()
    explorar_hiperparametros()