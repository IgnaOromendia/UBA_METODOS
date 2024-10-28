import numpy as np
import pandas as pd
from scipy import stats #type: ignore

mapMovieIndex = {"science fiction": 0, "romance":1, "crime":2, "western":3}
mapIndexMovie = {0:"science fiction", 1:"romance", 2:"crime", 3:"western"}

# Matriz de tokens
def matriz_tokens(Q, dataSet):
    tokens = np.hstack(df["tokens"].apply(lambda x: x.split()).values)

    unique_tokens = pd.Series(tokens).value_counts().index[:Q].values
    unique_tokens_dict = dict(zip(unique_tokens, range(len(unique_tokens))))

    X = np.zeros((len(dataSet), len(unique_tokens)), dtype=int)
    for i, row in dataSet.iterrows():
        for token in row["tokens"].split():
            if unique_tokens_dict.get(token,False)!=False:
                X[i, unique_tokens_dict[token]] += 1
    
    return X

def dist_coseno(X, Y):
    X_norm = np.linalg.norm(X, axis=1, ord=2, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, ord=2, keepdims=True)
    return 1 - (X @ Y.T) / (X_norm @ Y_norm.T)

def knn(i, k, D, trainMap):
    # Obtenemos los k vecinos más cercanos
    cercanos = np.argsort(D[i])[:k]

    # Ubicamos los id de género que elegimos
    ids = df.iloc[trainMap[cercanos]]["GenreID"].values

    # Devolvemos la moda de los géneros
    return mapIndexMovie[stats.mode(ids, keepdims=True).mode[0]]

def clasificar(k, D, testMap, trainMap):
    predict = {}

    # Guardamos un mapeo (indice original del df, predicción)
    for t in range(D.shape[0]):
        predict[testMap[t]] = knn(t, k, D, trainMap)

    return predict

def performance(predictions):
    acertados = 0
    
    # Contabilizamos los hits
    for i, predict in predictions.items():
        if df["Genre"][i] == predict:
            acertados += 1

    # Sacamos el promedio
    return acertados / len(predictions)

def clasificador_de_genero(Q, k):
    # Filtramos en trainning y testing reseteanod los índices para poder acceder a la matriz de distancias D
    train_set = df[df["split"] == "train"].reset_index()
    test_set  = df[df["split"] == "test"].reset_index()

    # Guardamos un mapeo para cada data set
    mapeoIndicesTrainSet = train_set["index"].to_numpy()
    mapeoIndicesTestSet  = test_set["index"].to_numpy()

    # Genermaos la matriz de tokens para ambos sets
    X_train = matriz_tokens(Q, train_set)
    X_test  = matriz_tokens(Q, test_set)    

    # Calulamos las distancias entre el trainning set y el testing set
    D = dist_coseno(X_test, X_train)

    # Clasificamos
    predictions = clasificar(k, D, mapeoIndicesTestSet, mapeoIndicesTrainSet)

    # Medimos la performance
    print(performance(predictions))

# Leer datos:                   
df = pd.read_csv("datos.csv")
df["GenreID"] = df["Genre"].apply(lambda x: mapMovieIndex[x])
clasificador_de_genero(Q=1000, k=5)
