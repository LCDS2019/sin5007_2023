import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from cursos_df import seed_value, X, y
from pipelines import pipelines
from pretty_print import prettify, ascii_as_image
from train_and_predict import train_and_predict

np.random.seed(seed_value)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)

train_and_predict_f = lambda p: train_and_predict(
    X_train=X_train,
    Y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value),
    param_grid=p[1]['params'],
    pipeline=p[1]['pipeline'],
)

result = list(map(train_and_predict_f, filter(lambda x: not x[1]['skip'], pipelines.items())))
df = pd.DataFrame(result)

ascii_table = prettify(df, drop_columns=['best_estimator'])
ascii_as_image(ascii_table, 'classifiers.jpg')
