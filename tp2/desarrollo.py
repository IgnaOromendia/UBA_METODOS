import numpy as np
import pandas as pd
from scipy import stats #type: ignore

mapMovieIndex = {"science fiction": 0, "romance":1, "crime":2, "western":3}
mapIndexMovie = {0:"science fiction", 1:"romance", 2:"crime", 3:"western"}

# Leer datos:                   
df = pd.read_csv("datos.csv")
df["GenreID"] = df["Genre"].apply(lambda x: mapMovieIndex[x])

# Matriz de tokens
def matriz_tokens(Q, dataSet):
    tokens = np.hstack(dataSet["tokens"].apply(lambda x: x.split()).values)

    unique_tokens = pd.Series(tokens).value_counts().index[:Q].values
    unique_tokens_dict = dict(zip(unique_tokens, range(len(unique_tokens))))

    X = np.zeros((len(dataSet), len(unique_tokens)), dtype=int)
    for i, row in dataSet.iterrows():
        for token in row["tokens"].split():
            if unique_tokens_dict.get(token,False)!=False:
                X[i, unique_tokens_dict[token]] += 1
    
    return X

def mat_covarianza(X):
    n = X.shape[0]
    return (X @ X.T) / (n-1)

def dist_coseno(X,Y):
    return 1 - ((X @ Y.T) / ((np.linalg.norm(X, ord=2) * (np.linalg.norm(Y, ord=2)))))

def knn(i, k, C, dataSet):
    cercanos = np.argsort(C[i])[::-1][:k]
    ids = np.array(df["GenreID"].values[cercanos])
    return mapIndexMovie[stats.mode(ids).mode]

def clasificar(k, D, dataSet):
    predict = {}

    for t in range(len(D)):
        predict[t] = knn(t, k, D, dataSet)

    return predict

# Probar con varios k como experimento !!!

def clasificador_de_genero(Q):
    k = 5

    train_set = df[df["split"] == "train"].dropna(subset=["tokens"]).reset_index(drop=True)
    test_set = df[df["split"] == "test"].dropna(subset=["tokens"]).reset_index(drop=True)

    X_train = matriz_tokens(Q, train_set)
    X_test  = matriz_tokens(Q, test_set)    

    distancias = dist_coseno(X_test, X_train)

    predictions = clasificar(k, distancias, test_set)

    print(performance(predictions, test_set))


def performance(predictions, dataSet):
    acertados = 0
    
    for (i, predict) in predictions.items():
        if dataSet["Genre"][i] == predict:
            acertados += 1

    return acertados / len(predictions)

clasificador_de_genero(1000)