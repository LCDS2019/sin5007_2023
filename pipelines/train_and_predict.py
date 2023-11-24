from time import time

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.utils import resample


def train_and_predict(X_train, X_test, y_train, y_test, pipeline, param_grid, cv):
    start_time = time()
    classifier_name = pipeline.steps[-1][1].__class__.__name__

    positive_class = 'Pública'

    precision_scorer = make_scorer(precision_score, pos_label=positive_class)
    recall_scorer = make_scorer(recall_score, pos_label=positive_class)
    f1_scorer = make_scorer(f1_score, pos_label=positive_class)

    # Configura e executa o GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring={
            'accuracy': 'accuracy',
            'f1': f1_scorer,
            'recall': recall_scorer,
            'precision': precision_scorer
        },
        refit='accuracy',
        n_jobs=-1,
        verbose=3
    )

    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_

    report_summary = resample_and_predict(best_estimator, X_test, y_test)

    end_time = time()

    return {
        'model': {'classifier': classifier_name, 'steps': list(pipeline.named_steps)},
        'best_params': grid_search.best_params_,
        'report': report_summary,
        'validation_report': extract_validation_data(grid_search.cv_results_),
        'total_time': end_time - start_time,
        'best_estimator': best_estimator,
        'all_results': grid_search.cv_results_
    }


def extract_validation_data(cv_results):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    result = {'mean': {}, 'std': {}}

    for metric in metrics:
        mean_key = f'mean_test_{metric}'
        std_key = f'std_test_{metric}'
        if mean_key in cv_results:
            result['mean'][metric] = cv_results[mean_key].max()
            result['std'][metric] = cv_results[std_key][cv_results[mean_key].argmax()]

    return result


def resample_and_predict(estimator, X_test, y_test, sample_per_bootstrap=10000, n_bootstraps=100):
    scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for _ in range(n_bootstraps):
        X_bootstrap, y_bootstrap = resample(X_test, y_test, n_samples=sample_per_bootstrap)

        y_pred = estimator.predict(X_bootstrap)

        scores['accuracy'].append(accuracy_score(y_bootstrap, y_pred))
        scores['precision'].append(precision_score(y_bootstrap, y_pred, average='macro'))
        scores['recall'].append(recall_score(y_bootstrap, y_pred, average='macro'))
        scores['f1'].append(f1_score(y_bootstrap, y_pred, average='macro'))

    return {
        'mean': {metric: np.mean(scores[metric]) for metric in scores},
        'std': {metric: np.std(scores[metric]) for metric in scores}
    }
