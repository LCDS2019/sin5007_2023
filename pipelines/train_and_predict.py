from time import time

import numpy as np
from sklearn.metrics import confusion_matrix
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

    scores_precision = cross_val_score(best_estimator, X_train, Y_train, cv=cv, scoring='precision_macro')
    scores_recall = cross_val_score(best_estimator, X_train, Y_train, cv=cv, scoring='recall_macro')
    scores_f1 = cross_val_score(best_estimator, X_train, Y_train, cv=cv, scoring='f1_macro')

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
            'precision': np.mean(scores_precision),
            'sensitivity': np.mean(scores_recall),
            'f1_score': np.mean(scores_f1),
            # 'specificity': avg_specificity,
        },
        'std': {
            'accuracy': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
            'precision': np.std(scores_precision),
            'sensitivity': np.std(scores_recall),
            'f1_score': np.std(scores_f1),
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
