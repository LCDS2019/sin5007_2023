########################################################################
print('')
print(f''.ljust(80 ,'#'))
print(f'Sin5007 - Reconhecimento de padrões')
print(f'Atividade 05 - cross validation')
print(f'Prof. MSc. Leonardo Santos')
print(f''.ljust(80 ,'#'))
print('')

########################################################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer,accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

#random_state
rs=42

########################################################################
print(f' Import de dados '.center(80 ,'#'))
print('')

#Total
df = pd.read_csv('arquivos/dados/tema02_cursos_2021_ti_02_encoded.csv',index_col=0,sep='|')

#PCA
#df = pd.read_csv('arquivos/dados/tema02_cursos_2021_ti_04_pca.csv',index_col=0,sep='|')

#Relief
#df = pd.read_csv('arquivos/dados/tema02_cursos_2021_ti_02_encoded.csv',index_col=0,sep='|')
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

#RUS
#df = pd.read_csv('arquivos/dados/tema05_df_rus.csv',index_col=0,sep='|')

#ROS
#
#df = pd.read_csv('arquivos/dados/tema05_df_ros.csv',index_col=0,sep='|')

X=df.drop(['TP_REDE_DEPARA'],axis=1)
y=df['TP_REDE_DEPARA']

print(f'Shape:{df.shape}')
print(f'X-Shape:{X.shape}')
print(f'y-Shape:{y.shape}')
print('')

########################################################################
print(f' Modelo '.center(80 ,'#'))
print('')

modelo = GaussianNB()
print(modelo)
print('')

param_grid = {
    'var_smoothing': np.logspace(0,-15, num=100)
}

'''
'var_smoothing': np.logspace(0,-9, num=100)
np.logspace retorna números espaçados uniformemente em uma escala logarítmica, 
começa em 0, termina em -9 e gera 100 amostras.
'''

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score),
    'f1': make_scorer(f1_score)
}

########################################################################
print(f' Cross-validation '.center(80 ,'#'))
print('')

k=5
k_val=1/(k-1)

skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rs)

grid_search = GridSearchCV(modelo, param_grid, cv=skf, scoring=scoring, refit='f1', n_jobs=-1)

accuracy_scores = []
recall_scores = []
precision_scores = []
f1_scores = []

test_accuracy_scores = []
test_recall_scores = []
test_precision_scores = []
test_f1_scores = []

parametros=[]

i=1
for train_val_index, test_index in skf.split(X, y):
    print(f'Loop {i} '.ljust(60, '*'))

    train_val=df.iloc[train_val_index]
    X_train_val = train_val.drop(['TP_REDE_DEPARA'], axis=1)
    y_train_val = train_val['TP_REDE_DEPARA']
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=k_val, random_state=rs)

    test = df.iloc[test_index]
    X_test = test.drop(['TP_REDE_DEPARA'], axis=1)
    y_test = test['TP_REDE_DEPARA']

    y_train = [1 if label == 'Privada' else 0 for label in y_train]
    y_val = [1 if label == 'Privada' else 0 for label in y_val]
    y_test = [1 if label == 'Privada' else 0 for label in y_test]

    #normalização
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print('')
    #print(' Train_val:', len(train_val_index))
    print(' Train:', len(X_train))
    print(' Val:', len(X_val))
    print(' Test:', len(test_index))
    print('')

    ##########################################################################################

    #from imblearn.under_sampling import RandomUnderSampler
    #rus = RandomUnderSampler(sampling_strategy='auto')
    #X_train, y_train = rus.fit_resample(X_train, y_train)

    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(sampling_strategy='auto')
    X_train, y_train = ros.fit_resample(X_train, y_train)

    print(len(y_train))

    ##########################################################################################

    grid_search.fit(X_train, y_train)
    melhor_modelo = grid_search.best_estimator_

    y_val_pred = melhor_modelo.predict(X_val)

    accuracy = accuracy_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    print('Model - valuation')
    print(f' accuracy: {accuracy:.4f}')
    print(f' recall: {recall:.4f}')
    print(f' precision: {precision:.4f}')
    print(f' f1_score: {f1:.4f}')

    print(grid_search.best_params_)
    parametros.append(grid_search.best_params_['var_smoothing'])

    print('')

    accuracy_scores.append(accuracy)
    recall_scores.append(recall)
    precision_scores.append(precision)
    f1_scores.append(f1)

    y_test_pred = melhor_modelo.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print('Model - Test')
    print(f' f1_score: {test_f1:.4f}')
    print(f' accuracy: {test_accuracy:.4f}')
    print(f' recall: {test_recall:.4f}')
    print(f' precision: {test_precision:.4f}')

    print('')

    test_accuracy_scores.append(test_accuracy)
    test_recall_scores.append(test_recall)
    test_precision_scores.append(test_precision)
    test_f1_scores.append(test_f1)

    confusion = confusion_matrix(y_test, y_test_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predições')
    plt.ylabel('Valores Reais')
    plt.title(f'Matriz de Confusão (0 = Pública / 1 = Privada ) - Loop: {i}')
    #plt.show()

    i=i+1

########################################################################
print(f' Report '.center(80 ,'#'))
print('')

print(f' Model - Valuation '.center(60, '*'))
accuracy_media = np.mean(accuracy_scores)
recall_media = np.mean(recall_scores)
precision_media = np.mean(precision_scores)
f1_media = np.mean(f1_scores)

print(f'Médias')
print(f'F1-score Médio: {f1_media:.4f}')
print(f'Acurácia Média: {accuracy_media:.4f}')
print(f'Revocação Média: {recall_media:.4f}')
print(f'Precisão Média: {precision_media:.4f}')

accuracy_desvio_padrao = np.std(accuracy_scores)
recall_desvio_padrao = np.std(recall_scores)
precision_desvio_padrao = np.std(precision_scores)
f1_desvio_padrao = np.std(f1_scores)

print('')
print(f'Desvios Padrão')
print(f'Desvio Padrão do F1-score: {f1_desvio_padrao:.4f}')
print(f'Desvio Padrão da Acurácia: {accuracy_desvio_padrao:.4f}')
print(f'Desvio Padrão da Revocação: {recall_desvio_padrao:.4f}')
print(f'Desvio Padrão da Precisão: {precision_desvio_padrao:.4f}')

#print(parametros)
parametros_media = np.mean(parametros)
print(f'Best params average: {parametros_media:.1e}')

print(f' Model - Test '.center(60, '*'))

test_accuracy_media = np.mean(test_accuracy_scores)
test_recall_media = np.mean(test_recall_scores)
test_precision_media = np.mean(test_precision_scores)
test_f1_media = np.mean(test_f1_scores)

print(f'Médias')
print(f'F1-score Médio: {test_f1_media:.4f}')
print(f'Acurácia Média: {test_accuracy_media:.4f}')
print(f'Revocação Média: {test_recall_media:.4f}')
print(f'Precisão Média: {test_precision_media:.4f}')

test_accuracy_desvio_padrao = np.std(test_accuracy_scores)
test_recall_desvio_padrao = np.std(test_recall_scores)
test_precision_desvio_padrao = np.std(test_precision_scores)
test_f1_desvio_padrao = np.std(test_f1_scores)

print('')
print(f'Desvios Padrão')
print(f'Desvio Padrão do F1-score: {test_f1_desvio_padrao:.4f}')
print(f'Desvio Padrão da Acurácia: {test_accuracy_desvio_padrao:.4f}')
print(f'Desvio Padrão da Revocação: {test_recall_desvio_padrao:.4f}')
print(f'Desvio Padrão da Precisão: {test_precision_desvio_padrao:.4f}')


