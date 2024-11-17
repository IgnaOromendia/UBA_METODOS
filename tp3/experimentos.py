import matplotlib.pyplot as plt
import desarrollo as ds

def plot_error_sujeto_1():
    grados, err_ajuste, err_val = ds.predecir_legrande(sujeto=1)

    plt.plot(grados, err_ajuste,'.-', label='Ajuste')
    plt.plot(grados, err_val,'.-', label='Val')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Grado')
    plt.ylabel('Error')
    plt.title('ECM de ajuste y validación')
    plt.savefig('graficos/grafico_error_sujeto1.png')
    plt.show()


if __name__ == "__main__":
    print("Experimentando...")
    plot_error_sujeto_1()