import numpy as np
import pandas as pd
from scipy import stats #type: ignore
import multiprocessing as mp
import subprocess as sp
import csv
import os

mapMovieIndex = {"science fiction": 0, "romance":1, "crime":2, "western":3}
mapIndexMovie = {0:"science fiction", 1:"romance", 2:"crime", 3:"western"}

### DATA SET
def leer_data_frame():
    df = pd.read_csv("datos.csv")
    df["GenreID"] = df["Genre"].apply(lambda x: mapMovieIndex[x])
    return df

### AUTOVALORES Y AUTOVECTORES

def escribir_input_experimento_mp(mat_a_escribir, archivo):
    with open(archivo, 'w') as f:
        f.write(str(len(mat_a_escribir)) + "\n")
        f.write(str(len(mat_a_escribir[0])) + "\n")

        for matrices in mat_a_escribir:
            for A in matrices:
                f.write(str(len(A)) + " " + str(len(A[0])) + "\n")
                for i in range(len(A)):
                    for j in range(len(A[0])):
                        f.write(str(A[i][j]) + " ")
                    f.write("\n")

def escribir_input_mp(matrices, archivo):
    with open(archivo, 'w') as f:
        f.write(str(len(matrices)) + "\n")
        
        for A in matrices:
            f.write(str(A.shape[0]) + " " + str(A.shape[1]) + "\n")
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    f.write(str(A[i][j]) + " ")
                f.write("\n")

def leer_output_experimento_mp(fileName):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        
        next(reader) # Salteamos los headings

        mat_id = 1
        autovals = []
        autovecs = []
        it_prom = []
        it_des = []
        err_prom = []
        err_des = []
        datos_matrices = []

        for linea in reader:
            new_mat_id = int(linea[0])

            if mat_id != new_mat_id:
                datos_matrices.append((autovals.copy(), autovecs.copy(), it_prom.copy(), it_des.copy(), err_prom.copy(), err_des.copy()))
                mat_id = new_mat_id
                autovecs.clear()
                autovals.clear()
                it_prom.clear()
                it_des.clear()
                err_prom.clear()
                err_des.clear()
                
            autovals.append(float(linea[1]))
            autovecs.append(np.array([float(i) for i in linea[2:-4]]))
            it_prom.append(float(linea[-4]))
            it_des.append(float(linea[-3]))
            err_prom.append(float(linea[-2]))
            err_des.append(float(linea[-1]))

        datos_matrices.append((autovals.copy(), autovecs.copy(), it_prom.copy(), it_des.copy(), err_prom.copy(), err_des.copy())) # Agregamos los datos de la última matriz

    return datos_matrices

def leer_output_mp(fileName):
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        
        next(reader) # Salteamos los headings

        mat_id = 1
        autovals = []
        autovecs = []
        datos_matrices = []

        for linea in reader:
            new_mat_id = int(linea[0])

            if mat_id != new_mat_id:
                datos_matrices.append((autovals.copy(), autovecs.copy()))
                mat_id = new_mat_id
                autovecs.clear()
                autovals.clear()
                
            autovals.append(float(linea[1]))
            autovecs.append(np.array([float(i) for i in linea[2:-1]]))
            

        datos_matrices.append((autovals.copy(), autovecs.copy())) # Agregamos los datos de la última matriz

    return datos_matrices

def calular_autovalores(nombre_archivo, output_name, experimento):
    resultado = sp.run(['./ejecMP', nombre_archivo, output_name, experimento], text=True)
    if resultado.returncode == 0:
        print("Datos en " + output_name)
    else:
        print("Error:", resultado.stderr)

def separarAutoData(resultado):
    return np.array(resultado[0][0]), np.array(resultado[0][1])

### MATRICES AUXILIARES

def matriz_covarianza(X):
    return X.T @ X / (X.shape[0] - 1)

def matriz_tokens(Q, dataFrame):
    tokens = np.hstack(dataFrame["tokens"].apply(lambda x: x.split()).values)

    unique_tokens = pd.Series(tokens).value_counts().index[:Q].values
    unique_tokens_dict = dict(zip(unique_tokens, range(len(unique_tokens))))

    X = np.zeros((len(dataFrame), len(unique_tokens)), dtype=int)
    for i, row in dataFrame.iterrows():
        for token in row["tokens"].split():
            if unique_tokens_dict.get(token,False)!=False:
                X[i, unique_tokens_dict[token]] += 1

    return X

def promedio(lista):
    suma = 0.0
    for elem in lista:
        suma += elem
    
    return suma / len(lista)  

### KNN

def dist_coseno(X, Y):
    X_norm = np.linalg.norm(X, axis=1, ord=2, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, ord=2, keepdims=True)

    Xn = X / X_norm
    Yn = Y / Y_norm

    return 1 - Xn @ Yn.T

def knn(k, X_train, X_test, generos_train):
    D = dist_coseno(X_train, X_test)

    # Obtenemos los k vecinos más cercanos
    cercanos = np.argsort(D, axis=0)[:k]
    
    # Ubicamos los id de género que elegimos
    generos = generos_train[cercanos]

    # Devolvemos la moda de los géneros
    return stats.mode(generos, keepdims=True).mode[0]

def performance(predicciones, generos_test):
    acertados = 0
    
    # Contabilizamos los hits
    for i, predict in enumerate(predicciones):
        if generos_test[i] == predict:
            acertados += 1

    # Sacamos el promedio
    return acertados / len(predicciones)

def clasificador_de_genero(k, X, train_set, test_set):
    # Guardamos un mapeo para cada data set
    generos_train = train_set["GenreID"].to_numpy()
    generos_test  = test_set["GenreID"].to_numpy()

    # Genermaos la matriz de tokens para ambos sets
    X_train = X[train_set.index]
    X_test  = X[test_set.index]

    # Clasificamos
    predicciones = knn(k, X_train, X_test, generos_train)

    # Medimos la performance
    return performance(predicciones, generos_test)

### KNN CROSS VALIDATION
    
def knn_cross_validation(k, X, folds, dataFrame):
    dataFrame["Partition"] = [i % folds for i in range(len(dataFrame))]

    performances = []
    
    for i in range(folds):
        test_set  = dataFrame[dataFrame["Partition"] == i]
        train_set = dataFrame[dataFrame["Partition"] != i]

        performances.append(clasificador_de_genero(k, X, train_set, test_set))

    return promedio(performances) 

def explorar_parametro_k(X, folds, dataFrame):
    max_k = 1
    max_perf = 0
    ult_act = 0

    for k in range(1, dataFrame.shape[0]):

        perf_k = knn_cross_validation(k, X, folds, dataFrame)

        if abs(ult_act - k) > 100: break

        if perf_k > max_perf:
            ult_act = k
            max_k = k
            max_perf = perf_k

    return max_k, max_perf

### PCA

def pca(X, id_matriz):
    C_train = matriz_covarianza(X)

    # Calcular autovalores
    output_file = "pca_output_" + str(id_matriz) + ".csv"

    # En caso de que ya esten calculados no los vuelva a calcular
    if not os.path.exists(output_file) or os.stat(output_file).st_size == 0:
        input_file = "pca_input.dat"
        escribir_input_mp([C_train],input_file)
        calular_autovalores(input_file, output_file, "0")

    w, V = separarAutoData(leer_output_mp(output_file))

    indices = np.argsort(w)[::-1]

    w = w[indices]
    V = V[:,indices]

    var = np.cumsum(w) / np.sum(w)

    return var, V

### PIPELINE FINAL

def fold_cross_validation(i, Q, X, train_df):
    desarrollo_set  = train_df[train_df["Partition"] == i]
    train_set       = train_df[train_df["Partition"] != i]

    generos_train       = train_set["GenreID"].to_numpy()
    generos_desarrollo  = desarrollo_set["GenreID"].to_numpy()

    X_train       = X[train_set.index]
    X_desarrollo  = X[desarrollo_set.index]

    X_train_c = X_train - X_train.mean(0)
    X_desarrollo_c = X_desarrollo - X_train.mean(0)

    var, V = pca(X_train_c, i)

    print("Comienza fold: " + str(i+1))

    resultados = np.zeros((Q,Q))

    for p in range(1, X_train.shape[0]):
        # Cambiamos de base para las p componentes principales
        X_train_hat      = X_train_c @ V[:, :p]
        X_desarrollo_hat = X_desarrollo_c @ V[:, :p]

        if var[p] < 0.95: continue
        
        ult_act = 0
        max_k = 0

        for k in range(1, train_df.shape[0]):
            predicciones = knn(k, X_train_hat, X_desarrollo_hat, generos_train)

            exa_k = performance(predicciones, generos_desarrollo)

            if abs(ult_act - k) > 100: break

            if max_k < exa_k:
                max_k = exa_k
                ult_act = k

            resultados[p, k] = exa_k

    print("Termina fold: " + str(i+1), exa_k)

    return resultados

def mejores_parametros(dataFrame, Q, X, folds):
    train_df = dataFrame[dataFrame["split"] == "train"].copy()
    train_df["Partition"] = [i % folds for i in range(len(train_df))]

    params = [(i, Q, X, train_df) for i in range(folds)]

    with mp.Pool(processes=folds) as pool:
        resultadosMP = pool.starmap(fold_cross_validation, params)

    resultados = np.zeros((Q,Q))

    for exaMat in resultadosMP:
        resultados += exaMat

    resultados = resultados / folds

    p, k = np.unravel_index(np.argmax(resultados), resultados.shape)

    return p, k

def pipeline_final(dataFrame, Q=1000, folds=4):
    X = matriz_tokens(Q, dataFrame)

    p, k = mejores_parametros(dataFrame, Q, X, folds)

    test_set  = dataFrame[dataFrame["split"] == "test"]
    train_set = dataFrame[dataFrame["split"] == "train"]

    generos_train = train_set["GenreID"].to_numpy()
    generos_test  = test_set["GenreID"].to_numpy()

    X_train = X[train_set.index]
    X_test  = X[test_set.index]

    X_train_c = X_train - X_train.mean(0)
    X_test_c = X_test - X_train.mean(0)
        
    var, V = pca(X_train_c, 'T')

    X_train_hat = X_train_c @ V[:,:p]
    X_test_hat  = X_test_c @ V[:,:p]

    predicciones = knn(k, X_train_hat, X_test_hat, generos_train)
    
    print("Mejor p:", p)
    print("Mejor k:", k)
    return performance(predicciones, generos_test)
