########################################################################
print('')
print(f''.ljust(80 ,'#'))
print(f'Sin5007 - Reconhecimento de padrões')
print(f'Atividade 04 - Classificação utilizando Naive Bayes Classifier - Total')
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