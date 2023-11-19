from time import time

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, cross_val_score


def train_and_predict(X_train, X_test, Y_train, y_test, pipeline, param_grid, cv):
    start_time = time()
    classifier_name = pipeline.steps[-1][1].__class__.__name__

    # Configura e executa o GridSearchCV
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1,
                               verbose=3)
    grid_search.fit(X_train, Y_train)

    # Previsões e métricas para o melhor modelo
    best_estimator = grid_search.best_estimator_

    y_pred = best_estimator.predict(X_test)

    # Precision
    precision_scores = cross_val_score(best_estimator, X_train, Y_train, cv=cv, scoring='precision_macro')
    precision_std = np.std(precision_scores)
    precision_mean = np.mean(precision_scores)

    # Recall
    recall_scores = cross_val_score(best_estimator, X_train, Y_train, cv=cv, scoring='recall_macro')
    recall_std = np.std(recall_scores)
    recall_mean = np.mean(recall_scores)

    # F1-Score
    f1_scores = cross_val_score(best_estimator, X_train, Y_train, cv=cv, scoring='f1_macro')
    f1_std = np.std(f1_scores)
    f1_mean = np.mean(f1_scores)

    # Calculando a especificidade manualmente
    cm = confusion_matrix(y_test, y_pred, labels=['Privada', 'Pública'])
    tn, fp, fn, tp = cm.ravel()
    specificity_privada = tn / (tn + fp)
    specificity_publica = tp / (tp + fn)
    avg_specificity = (specificity_privada + specificity_publica) / 2

    # Relatório final com desvios padrão
    report_summary = {
        'mean': {
            'accuracy': grid_search.best_score_,
            'precision': precision_mean,
            'sensitivity': recall_mean,
            'f1_score': f1_mean,
            # 'specificity': avg_specificity,
        },
        'std': {
            'accuracy': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
            'precision': precision_std,
            'sensitivity': recall_std,
            'f1_score': f1_std,
            # 'specificity': avg_specificity,
        }
    }

    end_time = time()

    return {
        'model': {'classifier': classifier_name, 'steps': list(pipeline.named_steps)},
        'best_params': grid_search.best_params_,
        'report': report_summary,
        'total_time': end_time - start_time,
        'best_estimator': best_estimator,
        'all_results': grid_search.cv_results_
    }
