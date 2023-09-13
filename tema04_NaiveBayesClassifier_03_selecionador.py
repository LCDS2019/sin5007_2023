########################################################################
print('')
print(f''.ljust(80 ,'#'))
print(f'Sin5007 - Reconhecimento de padrões')
print(f'Atividade 04 - Classificação utilizando Naive Bayes Classifier - Selecionador')
print(f'Prof. MSc. Leonardo Santos')
print(f''.ljust(80 ,'#'))
print('')

########################################################################
print(f' Carga de bibliotecas '.center(80 ,'#'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Defina a semente
seed_value = 45
np.random.seed(seed_value)

########################################################################
print(f' Carga de dados '.center(80 ,'#'))

df = pd.read_csv('arquivos/dados/tema02_cursos_2021_ti_03_normalized.csv',index_col=0,sep='|')

filtro22=['NO_UF_Rondônia','NO_CURSO_DEPARA_SComputação','NO_CURSO_DEPARA_Agrocomputação','NO_UF_Paraíba',
          'NO_UF_Tocantins','NO_UF_Rio Grande do Norte','NO_UF_Amazonas','NO_UF_Sergipe','NO_UF_Distrito Federal',
          'NO_UF_Acre','NO_UF_Amapá','NO_UF_Roraima',
          'TP_DIMENSAO_DEPARA_Cursos a distância com dimensão de dados somente a nível Brasil',
          'NO_CURSO_DEPARA_Eng. El. Ênfase Em Computação',
          'TP_ORGANIZACAO_ACADEMICA_DEPARA_Centro Federal de Educação Tecnológica',
          'TP_DIMENSAO_DEPARA_Cursos a distância ofertados por instituições brasileiras no exterior',
          'NO_UF_Sem_uf','NO_REGIAO_Sem_regiao','NO_CURSO_DEPARA_Mat. e CCientífica','NO_CURSO_DEPARA_Eng. de Sistemas',
          'TP_NIVEL_ACADEMICO_DEPARA_Sequencial de Formação Específica','TP_NIVEL_ACADEMICO_DEPARA_Graduação',
          'TP_REDE_DEPARA']

df=df[filtro22]

print(df.head())
print(df.shape)

X = df.drop('TP_REDE_DEPARA', axis=1)
y = df['TP_REDE_DEPARA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed_value)

########################################################################
print(f' Naive Bayes Classifier '.center(80 ,'#'))

# Inicialize o classificador Naive Bayes Gaussiano
naive_bayes_classifier = GaussianNB()

# Treine o classificador com os dados de treinamento
naive_bayes_classifier.fit(X_train, y_train)

# Faça previsões nos dados de teste
y_pred = naive_bayes_classifier.predict(X_test)

# Avalie a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo Naive Bayes:", accuracy)
