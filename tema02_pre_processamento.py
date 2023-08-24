########################################################################
print('')
print(f''.ljust(80 ,'#'))
print(f'Sin5007 - Reconhecimento de padrões')
print(f'Atividade 02 - Pré-processamento')
print(f'Prof. MSc. Leonardo Santos')
print(f''.ljust(80 ,'#'))
print('')

########################################################################
print(f' Carga de bibliotecas '.center(80 ,'#'))

import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

########################################################################
print(f' Carga de dados '.center(80 ,'#'))

df = pd.read_csv('arquivos/dados/tema01_cursos_2021_ti.csv',sep='|')
df = df.drop_duplicates()

print(f'Dimensões do conjunto de dados: {df.shape}')

df['NU_ANO_CENSO'] = df['NU_ANO_CENSO'].astype(str)
df['CO_MUNICIPIO'] = df['CO_MUNICIPIO'].astype(str)
df['CO_IES'] = df['CO_IES'].astype(str)
df['CO_MANTENEDORA'] = df['CO_MANTENEDORA'].astype(str)
df['CO_CURSO'] = df['CO_CURSO'].astype(str)

df['NO_REGIAO'] = df['NO_REGIAO'].apply(lambda x: 'Sem_regiao' if x == '0' else x)
df['NO_UF'] = df['NO_UF'].apply(lambda x: 'Sem_uf' if x == '0' else x)

df.to_csv('arquivos/dados/tema02_cursos_2021_ti_01.csv',sep='|', index=True)

print(df.head())
print(df.info())

########################################################################
print(f' Seleção de variáveis para o conjunto de dados final '.center(80 ,'#'))
df=df[[
'NO_REGIAO',
'NO_UF',
'IN_CAPITAL_DEPARA',
'NO_CURSO_DEPARA',
'TP_GRAU_ACADEMICO_DEPARA',
'IN_GRATUITO_DEPARA',
'TP_MODALIDADE_ENSINO_DEPARA',
'TP_NIVEL_ACADEMICO_DEPARA',
'TP_DIMENSAO_DEPARA',
'TP_ORGANIZACAO_ACADEMICA_DEPARA',
'TP_CATEGORIA_ADMINISTRATIVA_DEPARA',
'TP_REDE_DEPARA',
'QT_VG_TOTAL',
'QT_INSCRITO_TOTAL',
'QT_ING',
'QT_MAT',
'QT_CONC'
]]

print(df.info())


########################################################################
print(f' One-hot encode '.center(80 ,'#'))

categories = ['NO_REGIAO', 'NO_UF', 'IN_CAPITAL_DEPARA', 'NO_CURSO_DEPARA',
              'TP_GRAU_ACADEMICO_DEPARA','IN_GRATUITO_DEPARA','TP_MODALIDADE_ENSINO_DEPARA',
              'TP_NIVEL_ACADEMICO_DEPARA','TP_DIMENSAO_DEPARA','TP_ORGANIZACAO_ACADEMICA_DEPARA',
              'TP_CATEGORIA_ADMINISTRATIVA_DEPARA']

encoded_data = pd.get_dummies(df[categories], prefix=categories, prefix_sep='_')
encoded_data = encoded_data.apply(lambda x: x.astype(bool).astype(int))

encoded_data = encoded_data.merge(df[['QT_VG_TOTAL','QT_INSCRITO_TOTAL','QT_ING',
                                      'QT_MAT','QT_CONC','TP_REDE_DEPARA']],
                            left_index=True, right_index=True, how='left')

encoded_data.to_csv('arquivos/dados/tema02_cursos_2021_ti_02_encoded.csv',sep='|', index=True)
print(encoded_data)

########################################################################
print(f' Normalização de variáveis '.center(80 ,'#'))

scaler = StandardScaler()

columns=['QT_VG_TOTAL','QT_INSCRITO_TOTAL','QT_ING','QT_MAT','QT_CONC']

normalized_data = scaler.fit_transform(encoded_data[columns])
normalized_data = pd.DataFrame(normalized_data, columns=columns)
print(normalized_data)

########################################################################
print(f' merge de conjuntos de dados '.center(80 ,'#'))

columns_to_remove=['QT_VG_TOTAL','QT_INSCRITO_TOTAL','QT_ING','QT_MAT','QT_CONC','TP_REDE_DEPARA']
encoded_data_drop=encoded_data.drop(columns=columns_to_remove, axis=1)

normalized_data_vf = encoded_data_drop.merge(normalized_data,
                            left_index=True, right_index=True, how='left')

normalized_data_vf = normalized_data_vf.merge(encoded_data['TP_REDE_DEPARA'],
                            left_index=True, right_index=True, how='left')

normalized_data_vf.to_csv('arquivos/dados/tema02_cursos_2021_ti_03_normalized.csv',
                       sep='|', index=True)


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

principal_df_vf = principal_df.merge(encoded_data['TP_REDE_DEPARA'],
                            left_index=True, right_index=True, how='left')

principal_df_vf.to_csv('arquivos/dados/tema02_cursos_2021_ti_04_pca.csv',
                       sep='|', index=True)
print(principal_df_vf)