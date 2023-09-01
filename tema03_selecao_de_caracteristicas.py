########################################################################
print('')
print(f''.ljust(80 ,'#'))
print(f'Sin5007 - Reconhecimento de padrões')
print(f'Atividade 03 - seleção de características')
print(f'Prof. MSc. Leonardo Santos')
print(f''.ljust(80 ,'#'))
print('')

########################################################################
print(f' Carga de bibliotecas '.center(80 ,'#'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Defina a semente
seed_value = 50
np.random.seed(seed_value)

########################################################################
print(f' Carga de dados '.center(80 ,'#'))

df = pd.read_csv('arquivos/dados/tema02_cursos_2021_ti_03_normalized.csv',index_col=0,sep='|')

df=df[(df['TP_MODALIDADE_ENSINO_DEPARA_Presencial'] == 1) &
      ((df['NO_UF_São Paulo'] == 1))]

print(df.head())

X = df.drop('TP_REDE_DEPARA', axis=1)
y = df['TP_REDE_DEPARA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed_value)

csv=pd.DataFrame(y_test)
csv.to_csv('y_test.csv')
########################################################################
print(f' Criar um classificador de árvore de decisão '.center(80 ,'#'))

clf = DecisionTreeClassifier(random_state=seed_value)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)

cm = confusion_matrix(y_test, y_pred)

# Definir os nomes das categorias
class_names = ['Privada','Pública']

# Plotar a matriz de confusão usando Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 16})
plt.xlabel("Predicted")
plt.ylabel("True")

# Adicionar rótulos aos eixos
tick_marks = [0.5, 1.5]
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

plt.show()

#O elemento na linha 1, coluna 1 mostra o número de verdadeiros positivos (TP) - ou seja, os casos em que a classe positiva foi prevista corretamente.
#O elemento na linha 0, coluna 0 mostra o número de verdadeiros negativos (TN) - ou seja, os casos em que a classe negativa foi prevista corretamente.
#O elemento na linha 0, coluna 1 mostra o número de falsos positivos (FP) - ou seja, os casos em que a classe negativa foi erroneamente prevista como positiva.
#O elemento na linha 1, coluna 0 mostra o número de falsos negativos (FN) - ou seja, os casos em que a classe positiva foi erroneamente prevista como negativa.


# Gerar o relatório de classificação
target_names = ['Negative', 'Positive']
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

# Converter o relatório para um DataFrame do pandas para melhor apresentação
report_df = pd.DataFrame(report).transpose()

print("Relatório de Classificação:")
print(report_df)


'''
########################################################################
print(f' Seleção de atributos utilizando Algoritmos Genéticos '.center(80 ,'#'))

print(' Inicialização da População '.center(80 ,'-'))

um_vectors = 100
vector_size = 74

# Lista para armazenar os vetores gerados
vectors = []

# Loop para gerar os vetores
for _ in range(num_vectors):
    binary_vector = np.random.randint(0, 2, size=vector_size)
    vectors.append(binary_vector)










print(' Seleção '.center(80 ,'-'))

print(' Recombinação (Crossover) '.center(80 ,'-'))

print(' Mutação '.center(80 ,'-'))

print(' Atualização da População '.center(80 ,'-'))

print(' Critérios de Parada '.center(80 ,'-'))

print(' Resultado '.center(80 ,'-'))

'''