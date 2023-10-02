########################################################################
print('')
print(f''.ljust(80 ,'#'))
print(f'Sin5007 - Reconhecimento de padrões')
print(f'Atividade 05 - balanceamento do dataset')
print(f'Prof. MSc. Leonardo Santos')
print(f''.ljust(80 ,'#'))
print('')

########################################################################
print(f' Carga de bibliotecas '.center(80 ,'#'))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
########################################################################
print(f' Carga de dados '.center(80 ,'#'))

df = pd.read_csv('arquivos/dados/tema02_cursos_2021_ti_02_encoded.csv',index_col=0,sep='|')

print(df.head())

X=df.drop(['TP_REDE_DEPARA'],axis=1)
y=df['TP_REDE_DEPARA']

########################################################################
print(f' Conjunto de dados para o balanceamento '.center(80 ,'#'))
df_values=pd.DataFrame(y.value_counts())
print(df_values)

########################################################################
print(f' Balanceamento undersampling '.center(80 ,'#'))

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy=1)
X_rus,y_rus = rus.fit_resample(X,y)

df_values_rus=pd.DataFrame(y_rus.value_counts())
print(df_values_rus)

X_rus['TP_REDE_DEPARA'] = y_rus
df_rus=X_rus
print(f'Conjunto de dados undersampling: {df_rus.shape}')

df_rus.to_csv('arquivos/dados/tema05_df_rus.csv',sep='|', index=True)

########################################################################
print(f' Normalização de variáveis '.center(80 ,'#'))

scaler = StandardScaler()

columns=['QT_VG_TOTAL','QT_INSCRITO_TOTAL','QT_ING','QT_MAT','QT_CONC']

normalized_data = scaler.fit_transform(df_rus[columns])
normalized_data = pd.DataFrame(normalized_data, columns=columns)
print(normalized_data)

########################################################################
print(f' merge de conjuntos de dados '.center(80 ,'#'))

columns_to_remove=['QT_VG_TOTAL','QT_INSCRITO_TOTAL','QT_ING','QT_MAT','QT_CONC','TP_REDE_DEPARA']
encoded_data_drop=df_rus.drop(columns=columns_to_remove, axis=1)

encoded_data_drop=encoded_data_drop.reset_index(drop=True)
normalized_data=normalized_data.reset_index(drop=True)
df_rus=df_rus.reset_index(drop=True)

normalized_data_vf = encoded_data_drop.merge(normalized_data,
                            left_index=True, right_index=True, how='left')


normalized_data_vf = normalized_data_vf.merge(df_rus['TP_REDE_DEPARA'],
                            left_index=True, right_index=True, how='left')

normalized_data_vf.to_csv('arquivos/dados/tema05_df_rus_normalized.csv',sep='|', index=True)

print(normalized_data_vf)


########################################################################
print(f' PCA no conjunto de dados '.center(80 ,'#'))

data_X = normalized_data_vf.drop('TP_REDE_DEPARA', axis=1)

pca = PCA()
pca.fit(data_X)
explained_variance_ratio = pca.explained_variance_ratio_

# Plotar o gráfico de variância explicada acumulada
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Explicada Acumulada')
plt.title('Variância Explicada Acumulada pelo Número de Componentes Principais')
plt.show()

# Encontrar o número de componentes que captura uma quantidade específica de variância (por exemplo, 95%)
desired_variance = 0.95
num_components = np.argmax(cumulative_variance_ratio >= desired_variance) + 1
print(f'Número de Componentes Principais para {desired_variance:.0%} de Variância: {num_components}')

########################################################################
print(f' PCA com número de componentes esperado '.center(80 ,'#'))

# Definir o número de componentes principais desejado
num_components = num_components  # Altere para o número desejado

# Inicializar o PCA com o número de componentes especificado
pca = PCA(n_components=num_components)

# Aplicar o PCA aos dados
principal_components = pca.fit_transform(data_X)

# Criar um DataFrame com os componentes principais
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(num_components)])

# Imprimir os componentes principais
print(principal_df)

########################################################################
print(f' conjunto de dados final '.center(80 ,'#'))

principal_df_vf = principal_df.merge(normalized_data_vf['TP_REDE_DEPARA'],
                            left_index=True, right_index=True, how='left')

principal_df_vf.to_csv('arquivos/dados/tema05_df_rus_normalized_pca.csv',
                       sep='|', index=True)
print(principal_df_vf)


########################################################################
print(f' Relief - Carga de dados '.center(80 ,'#'))

df = pd.read_csv('arquivos/dados/tema05_df_rus.csv',index_col=0,sep='|')

#df=df[(df['TP_MODALIDADE_ENSINO_DEPARA_Presencial'] == 1) & (df['NO_CURSO_DEPARA_SInformação'] == 1)]

print(df.head())
print(df.shape)

X_rel = df.drop('TP_REDE_DEPARA', axis=1)
y_rel = df['TP_REDE_DEPARA']

########################################################################
print(f' Relief '.center(80 ,'#'))

def relief(X_rel, y_rel):
    X_rel = X_rel.to_numpy()
    y_rel = y_rel.to_numpy()
    print(f'Dimensões da entrada X: {X_rel.shape}')
    print(f'Dimensões da entrada y: {y_rel.shape}')

    num_samples, num_features = X_rel.shape
    weights = np.zeros(num_features)

    for i in range(num_samples):
        current_instance = X_rel[i,:]

        nearest_hit = None
        nearest_miss = None
        min_hit_distance = float("inf")
        min_miss_distance = float("inf")

        for j in range(num_samples):
            if i != j:
                distance = np.linalg.norm(current_instance - X_rel[j, :])
                if y_rel[i] == y_rel[j]:
                    if distance < min_hit_distance:
                        min_hit_distance = distance
                        nearest_hit = X_rel[j, :]
                else:
                    if distance < min_miss_distance:
                        min_miss_distance = distance
                        nearest_miss = X_rel[j, :]

        weights += np.abs(current_instance - nearest_hit) - np.abs(current_instance - nearest_miss)

    return weights / num_samples


# Calcular os pesos usando o algoritmo Relief
feature_weights = relief(X_rel, y_rel)

# Selecionar as características com maiores pesos
num_selected_features = 22
selected_features = np.argsort(feature_weights)[-num_selected_features:]

#print(selected_features)

feature_names = X.columns.tolist()

# Obter os nomes das características a partir dos índices
selected_feature_names = [feature_names[idx] for idx in selected_features]

print("Selected Feature Names:")

for i in selected_feature_names:
    print(i)

########################################################################
########################################################################
print(f' Balanceamento oversampling '.center(80 ,'#'))

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(sampling_strategy=1)
X_ros,y_ros = ros.fit_resample(X,y)

df_values_ros=pd.DataFrame(y_ros.value_counts())
print(df_values_ros)

X_ros['TP_REDE_DEPARA'] = y_ros
df_ros=X_ros
print(f'Conjunto de dados oversampling: {df_ros.shape}')

df_ros.to_csv('arquivos/dados/tema05_df_ros.csv',sep='|', index=True)

########################################################################
print(f' Normalização de variáveis '.center(80 ,'#'))

scaler = StandardScaler()

columns=['QT_VG_TOTAL','QT_INSCRITO_TOTAL','QT_ING','QT_MAT','QT_CONC']

normalized_data = scaler.fit_transform(df_ros[columns])
normalized_data = pd.DataFrame(normalized_data, columns=columns)
print(normalized_data)

########################################################################
print(f' merge de conjuntos de dados '.center(80 ,'#'))

columns_to_remove=['QT_VG_TOTAL','QT_INSCRITO_TOTAL','QT_ING','QT_MAT','QT_CONC','TP_REDE_DEPARA']
encoded_data_drop=df_ros.drop(columns=columns_to_remove, axis=1)

encoded_data_drop=encoded_data_drop.reset_index(drop=True)
normalized_data=normalized_data.reset_index(drop=True)
df_ros=df_ros.reset_index(drop=True)

normalized_data_vf = encoded_data_drop.merge(normalized_data,
                            left_index=True, right_index=True, how='left')


normalized_data_vf = normalized_data_vf.merge(df_ros['TP_REDE_DEPARA'],
                            left_index=True, right_index=True, how='left')

normalized_data_vf.to_csv('arquivos/dados/tema05_df_ros_normalized.csv',sep='|', index=True)

print(normalized_data_vf)


########################################################################
print(f' PCA no conjunto de dados '.center(80 ,'#'))

data_X = normalized_data_vf.drop('TP_REDE_DEPARA', axis=1)

pca = PCA()
pca.fit(data_X)
explained_variance_ratio = pca.explained_variance_ratio_

# Plotar o gráfico de variância explicada acumulada
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Explicada Acumulada')
plt.title('Variância Explicada Acumulada pelo Número de Componentes Principais')
plt.show()

# Encontrar o número de componentes que captura uma quantidade específica de variância (por exemplo, 95%)
desired_variance = 0.95
num_components = np.argmax(cumulative_variance_ratio >= desired_variance) + 1
print(f'Número de Componentes Principais para {desired_variance:.0%} de Variância: {num_components}')

########################################################################
print(f' PCA com número de componentes esperado '.center(80 ,'#'))

# Definir o número de componentes principais desejado
num_components = num_components  # Altere para o número desejado

# Inicializar o PCA com o número de componentes especificado
pca = PCA(n_components=num_components)

# Aplicar o PCA aos dados
principal_components = pca.fit_transform(data_X)

# Criar um DataFrame com os componentes principais
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(num_components)])

# Imprimir os componentes principais
print(principal_df)

########################################################################
print(f' conjunto de dados final '.center(80 ,'#'))

principal_df_vf = principal_df.merge(normalized_data_vf['TP_REDE_DEPARA'],
                            left_index=True, right_index=True, how='left')

principal_df_vf.to_csv('arquivos/dados/tema05_df_ros_normalized_pca.csv',
                       sep='|', index=True)
print(principal_df_vf)


########################################################################
print(f' Relief - Carga de dados '.center(80 ,'#'))

df = pd.read_csv('arquivos/dados/tema05_df_ros.csv',index_col=0,sep='|')

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
num_selected_features = 22
selected_features = np.argsort(feature_weights)[-num_selected_features:]

#print(selected_features)

feature_names = X.columns.tolist()

# Obter os nomes das características a partir dos índices
selected_feature_names = [feature_names[idx] for idx in selected_features]

print("Selected Feature Names:")

for i in selected_feature_names:
    print(i)