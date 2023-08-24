########################################################################
print('')
print(f''.ljust(80 ,'#'))
print(f'Sin5007 - Reconhecimento de padrões')
print(f'Atividade 01 - Análise descritiva')
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

print('\n'+'ok'+'\n')
########################################################################
print(f' Carga de dados '.center(80 ,'#'))
print('')

df = pd.read_csv('arquivos/dados/tema01_cursos_2021_ti.csv',sep='|')
print(f'Dimensões do conjunto de dados: {df.shape}')

df['NU_ANO_CENSO'] = df['NU_ANO_CENSO'].astype(str)
df['CO_MUNICIPIO'] = df['CO_MUNICIPIO'].astype(str)
df['CO_IES'] = df['CO_IES'].astype(str)
df['CO_MANTENEDORA'] = df['CO_MANTENEDORA'].astype(str)
df['CO_CURSO'] = df['CO_CURSO'].astype(str)


print(df.head())
print(df.info())

print(df.describe())

########################################################################
print(f' Análise de ingressantes '.center(80 ,'-'))

#1: 'Pública' 2: 'Privada'
rede = 'Pública'
#rede = 'Privada'

#1:'Presencial' 2:'Curso a distância'
modalidade = 'Presencial'
#modalidade = 'Curso a distância'


df_filtro = df[
    (df['TP_REDE_DEPARA']==rede) &
    (df['TP_MODALIDADE_ENSINO_DEPARA']==modalidade)]

########################################################################
plt.figure(figsize=(10, 6))  # Ajuste o tamanho da figura conforme necessário
sns.boxplot(data=df_filtro, x='NO_CURSO_DEPARA', y='QT_ING')
#plt.xlabel('Curso')
plt.ylabel('Número de Ingressantes')
plt.title(f'Distribuição de Ingressantes por Curso | rede {rede} | {modalidade}')
plt.xticks(rotation=45)
plt.tight_layout()

nome_arquivo=f'img/QT_ING_{rede}_{modalidade}.svg'
plt.savefig(nome_arquivo)
plt.show()
########################################################################
plt.figure(figsize=(10, 6))  # Ajuste o tamanho da figura conforme necessário
sns.boxplot(data=df_filtro, x='NO_CURSO_DEPARA', y='QT_MAT')
#plt.xlabel('Curso')
plt.ylabel('Número de Matriculados')
plt.title(f'Distribuição de Matriculados por Curso | rede {rede} | {modalidade}')
plt.xticks(rotation=45)
plt.tight_layout()

nome_arquivo=f'img/QT_MAT_{rede}_{modalidade}.svg'
plt.savefig(nome_arquivo)
plt.show()
########################################################################
plt.figure(figsize=(10, 6))  # Ajuste o tamanho da figura conforme necessário
sns.boxplot(data=df_filtro, x='NO_CURSO_DEPARA', y='QT_CONC')
#plt.xlabel('Curso')
plt.ylabel('Número de Concluintes')
plt.title(f'Distribuição de Concluintes por Curso | rede {rede} | {modalidade}')
plt.xticks(rotation=45)
plt.tight_layout()

nome_arquivo=f'img/QT_CONC_{rede}_{modalidade}.svg'
plt.savefig(nome_arquivo)
plt.show()
########################################################################


correlation_matrix = df_filtro[['QT_VG_TOTAL','QT_INSCRITO_TOTAL','QT_ING','QT_MAT','QT_CONC']].corr()

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.title('Correlações - matriz')
plt.colorbar()
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)

# Adicionando anotações de correlação aos elementos da matriz
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                 ha='center', va='center', color='white')

nome_arquivo=f'img/matrix_correl_{rede}_{modalidade}.svg'
plt.savefig(nome_arquivo)
plt.show()


plt.title('Correlações - pairplot')
sns.pairplot(data=df_filtro[['QT_VG_TOTAL','QT_INSCRITO_TOTAL','QT_ING','QT_MAT','QT_CONC']])
nome_arquivo=f'img/pairplot_{rede}_{modalidade}.svg'
plt.savefig(nome_arquivo)
plt.show()
