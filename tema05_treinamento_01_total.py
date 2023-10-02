########################################################################
print('')
print(f''.ljust(80 ,'#'))
print(f'Sin5007 - Reconhecimento de padrões')
print(f'Atividade 05 - treinamento 01 - conjunto total')
print(f'Prof. MSc. Leonardo Santos')
print(f''.ljust(80 ,'#'))
print('')

########################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import make_scorer,accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns

#random_state
rs=42

#kfolds
n_folds=10

########################################################################
print(f' Carga de dados '.center(80 ,'#'))

#df = pd.read_csv('arquivos/dados/tema02_cursos_2021_ti_02_encoded.csv',index_col=0,sep='|')
#df = pd.read_csv('arquivos/dados/tema02_cursos_2021_ti_04_pca.csv',index_col=0,sep='|')

#df = pd.read_csv('arquivos/dados/tema02_cursos_2021_ti_03_normalized.csv',index_col=0,sep='|')
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
#df=df[filtro22]

#df = pd.read_csv('arquivos/dados/tema05_df_rus.csv',index_col=0,sep='|')
df = pd.read_csv('arquivos/dados/tema05_df_ros.csv',index_col=0,sep='|')

X=df.drop(['TP_REDE_DEPARA'],axis=1)
y=df['TP_REDE_DEPARA']

print(df.head())
print(f'Shape:{df.shape}')

########################################################################
print(f' Divisão em treinamento, validação e teste  '.center(80 ,'#'))

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.25, random_state=rs)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.30, random_state=rs)

print(f'X:{X.shape}')
print(f'y:{y.shape}'+'\n')

print(f'X_train:{X_train.shape}')
print(f'y_train:{y_train.shape}'+'\n')
#print(X_train.head())
#print(y_train.head())
#print('\n')

print(f'X_val:{X_val.shape}')
print(f'y_val:{y_val.shape}'+'\n')
#print(X_val.head())
#print(y_val.head())
#print('\n')

print(f'X_test:{X_test.shape}')
print(f'y_test:{y_test.shape}'+'\n')
#print(X_test.head())
#print(y_test.head())
#print('\n')

########################################################################
print(f' Treinamento, validação e teste '.center(80 ,'#'))

accuracy_scores = []
recall_scores = []
precision_scores = []
f1_scores = []

modelo = GaussianNB()

param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7]
}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score),
    'f1': make_scorer(f1_score)
}

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
print(skf)

grid_search = GridSearchCV(modelo, param_grid, cv=skf, scoring=scoring, refit='f1')

for train_index, val_index in skf.split(X_train_val, y_train_val):
    X_train_fold, X_val_fold = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
    y_train_fold, y_val_fold = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

    grid_search.fit(X_train_fold, y_train_fold)
    melhor_modelo = grid_search.best_estimator_

    y_pred_fold = melhor_modelo.predict(X_val_fold)

    y_val_fold = [1 if label == 'Privada' else 0 for label in y_val_fold]
    y_pred_fold = [1 if label == 'Privada' else 0 for label in y_pred_fold]

    accuracy = accuracy_score(y_val_fold, y_pred_fold)
    recall = recall_score(y_val_fold, y_pred_fold)
    precision = precision_score(y_val_fold, y_pred_fold)
    f1 = f1_score(y_val_fold, y_pred_fold)

    accuracy_scores.append(accuracy)
    recall_scores.append(recall)
    precision_scores.append(precision)
    f1_scores.append(f1)

accuracy_media = np.mean(accuracy_scores)
recall_media = np.mean(recall_scores)
precision_media = np.mean(precision_scores)
f1_media = np.mean(f1_scores)

accuracy_desvio_padrao = np.std(accuracy_scores)
recall_desvio_padrao = np.std(recall_scores)
precision_desvio_padrao = np.std(precision_scores)
f1_desvio_padrao = np.std(f1_scores)

# Apresente as métricas médias e os desvios padrão
print("Métricas Médias:")
print("Acurácia Média:", accuracy_media)
print("Revocação Média:", recall_media)
print("Precisão Média:", precision_media)
print("F1-score Médio:", f1_media)

print('\n')
print("Desvios Padrão das Métricas:")
print("Desvio Padrão da Acurácia:", accuracy_desvio_padrao)
print("Desvio Padrão da Revocação:", recall_desvio_padrao)
print("Desvio Padrão da Precisão:", precision_desvio_padrao)
print("Desvio Padrão do F1-score:", f1_desvio_padrao)

print("Melhor Hiperparâmetro:")
print(grid_search.best_params_)

#########################################################################################
#modelo final

var_smoothing = grid_search.best_params_['var_smoothing']

nb_model = GaussianNB(var_smoothing=var_smoothing)
nb_model.fit(X_train_val, y_train_val)

y_pred_test = nb_model.predict(X_test)



y_test = [1 if label == 'Privada' else 0 for label in y_test]
y_pred_test = [1 if label == 'Privada' else 0 for label in y_pred_test]

accuracy_teste = accuracy_score(y_test, y_pred_test)
recall_teste = recall_score(y_test, y_pred_test)
precision_teste = precision_score(y_test, y_pred_test)
f1_teste = f1_score(y_test, y_pred_test)

confusion = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predições')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão (0 = Pública / 1 = Privada )')
plt.show()

print('\n')
print("Métricas no Conjunto de Teste:")
print("Acurácia no Teste:", accuracy_teste)
print("Revocação no Teste:", recall_teste)
print("Precisão no Teste:", precision_teste)
print("F1-score no Teste:", f1_teste)


print(X_test.shape)
