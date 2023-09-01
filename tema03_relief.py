########################################################################
print('')
print(f''.ljust(80 ,'#'))
print(f'Sin5007 - Reconhecimento de padrões')
print(f'Atividade 03 - seleção de características - Relief')
print(f'Prof. MSc. Leonardo Santos')
print(f''.ljust(80 ,'#'))
print('')

########################################################################
print(f' Carga de bibliotecas '.center(80 ,'#'))

import pandas as pd
import numpy as np

# Defina a semente
seed_value = 45
np.random.seed(seed_value)

########################################################################
print(f' Carga de dados '.center(80 ,'#'))

df = pd.read_csv('arquivos/dados/tema02_cursos_2021_ti_03_normalized.csv',index_col=0,sep='|')

#df=df[(df['TP_MODALIDADE_ENSINO_DEPARA_Presencial'] == 1) & (df['NO_CURSO_DEPARA_SInformação'] == 1)]

print(df.head())
print(df.shape)

X = df.drop('TP_REDE_DEPARA', axis=1)
y = df['TP_REDE_DEPARA']

########################################################################
print(f' Relief '.center(80 ,'#'))

def relief(X, y):
    X = X.to_numpy()
    y = y.to_numpy()
    print(f'Dimensões da entrada X: {X.shape}')
    print(f'Dimensões da entrada y: {y.shape}')

    num_samples, num_features = X.shape
    weights = np.zeros(num_features)

    for i in range(num_samples):
        current_instance = X[i,:]

        nearest_hit = None
        nearest_miss = None
        min_hit_distance = float("inf")
        min_miss_distance = float("inf")

        for j in range(num_samples):
            if i != j:
                distance = np.linalg.norm(current_instance - X[j, :])
                if y[i] == y[j]:
                    if distance < min_hit_distance:
                        min_hit_distance = distance
                        nearest_hit = X[j, :]
                else:
                    if distance < min_miss_distance:
                        min_miss_distance = distance
                        nearest_miss = X[j, :]

        weights += np.abs(current_instance - nearest_hit) - np.abs(current_instance - nearest_miss)

    return weights / num_samples


# Calcular os pesos usando o algoritmo Relief
feature_weights = relief(X, y)

# Selecionar as características com maiores pesos
num_selected_features = 20
selected_features = np.argsort(feature_weights)[-num_selected_features:]

#print(selected_features)

feature_names = X.columns.tolist()

# Obter os nomes das características a partir dos índices
selected_feature_names = [feature_names[idx] for idx in selected_features]

print("Selected Feature Names:")

for i in selected_feature_names:
    print(i)