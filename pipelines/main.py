import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from cursos_df import seed_value, X, y
from pipelines import pipelines
from pretty_print import prettify, ascii_as_image, plot_model_metrics
from train_and_predict import train_and_predict

np.random.seed(seed_value)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)

result = []

for pipeline in pipelines:
    result.append(
        train_and_predict(
            X_train=X_train,
            Y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value),
            param_grid=pipeline['params'],
            pipeline=pipeline['pipeline'],
        )
    )

plot_model_metrics(result)
df = pd.DataFrame(result)

ascii_table = prettify(df, drop_columns=['best_estimator', 'all_results'])
ascii_as_image(ascii_table, 'classifiers.jpg')

cv_results = pd.DataFrame(df['all_results'])

# Get the number of splits (k in k-fold)
n_splits = sum(1 for key in cv_results if key.startswith("split") and key.endswith("test_score"))

# Get the number of parameter combinations
n_combinations = len(cv_results)
