from time import time

from pandas import DataFrame
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, cross_val_predict


def train_and_predict(X_train, X_test, Y_train, y_test, pipeline, param_grid, cv):
    start_time = time()
    classifier_name = pipeline.steps[-1][1].__class__.__name__
    step_names = list(map(lambda s: s, pipeline.named_steps))

    # Configura o GridSearchCV com o pipeline e o conjunto de parâmetros fornecidos
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               cv=cv,
                               scoring='accuracy',
                               n_jobs=-1,
                               return_train_score=True,
                               verbose=3)

    # Fit via GridSearchCV
    grid_search.fit(X_train, Y_train)

    # Métricas para o melhor modelo
    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    n_fits = len(grid_search.cv_results_['params']) * cv.n_splits

    # Previsões de validação cruzada para o melhor estimador
    y_pred = cross_val_predict(best_estimator, X_test, y_test)

    # Calculando métricas adicionais
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=['Privada', 'Pública'],
                                                                     average=None)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity_privada = tn / (tn + fp)
    specificity_publica = tp / (tp + fn)

    average_precision = (precision[0] + precision[1]) / 2
    average_recall = (recall[0] + recall[1]) / 2
    average_f1_score = (f1[0] + f1[1]) / 2
    average_specificity = (specificity_privada + specificity_publica) / 2

    report = [
        {
            'class': 'privada',
            'precision': precision[0],
            'sensitivity': recall[0],
            'f1_score': f1[0],
            'specificity': specificity_privada,
            'instance_count': support[0]
        },
        {
            'class': 'publica',
            'precision': precision[1],
            'sensitivity': recall[1],
            'f1_score': f1[1],
            'specificity': specificity_publica,
            'instance_count': support[1]
        },
        {
            'class': 'privada + publica',
            'precision': average_precision,
            'sensitivity': average_recall,
            'f1_score': average_f1_score,
            'specificity': average_specificity,
            'instance_count': support[0] + support[1]
        }
    ]

    end_time = time()
    total_time = end_time - start_time

    return {
        'model': {
            'classifier': classifier_name,
            'steps': step_names
        },
        'best_score': best_score,
        'best_params': best_params,
        'accuracy': accuracy,
        'report': DataFrame(report),
        'n_fits': n_fits,
        'total_time': total_time,
        'best_estimator': best_estimator,
        'all_results': grid_search.cv_results_
    }
