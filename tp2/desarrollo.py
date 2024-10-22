import numpy as np
import pandas as pd
from scipy import stats #type: ignore

mapMovieIndex = {"science fiction": 0, "romance":1, "crime":2, "western":3}
mapIndexMovie = {0:"science fiction", 1:"romance", 2:"crime", 3:"western"}

# Leer datos:                   
df = pd.read_csv("datos.csv")

df["GenreID"] = df["Genre"].apply(lambda x: mapMovieIndex[x])

# Matriz de tokens
def matriz_tokens():
    tokens = np.hstack(df["tokens"].apply(lambda x: x.split()).values)

    unique_tokens = pd.Series(tokens).value_counts().index[:1000].values
    unique_tokens_dict = dict(zip(unique_tokens, range(len(unique_tokens))))
    X = np.zeros((len(df), len(unique_tokens)), dtype=int)
    for i, row in df.iterrows():
        for token in row["tokens"].split():
            if unique_tokens_dict.get(token,False)!=False:
                X[i, unique_tokens_dict[token]] += 1
    
    return X

def distancias(X):
    n = X.shape[0]
    return (X @ X.T) / (n-1)

def knn(i, k, C):
    cercanos = np.argsort(C[i])[::-1][:k]
    ids = np.array(df["GenreID"].values[cercanos])
    return mapIndexMovie[stats.mode(ids).mode]

def clasificar(k, C):
    predict = {}

    for i in range(len(C)):
        predict[i] = knn(i, k, C)

    return predict

def performance(predictions):
    acertados = 0
    
    for (i, predict) in predictions.items():
        if df["Genre"][i] == predict:
            acertados += 1

    return acertados / len(predictions)

# Probar con varios k como experimento
predictions = clasificar(25, distancias(matriz_tokens()))
print(performance(predictions))

