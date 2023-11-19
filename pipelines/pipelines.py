from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from scikeras.wrappers import KerasClassifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
import tensorflow as tf
from cursos_df import seed_value, X


def to_dense_f(x):
    return x.toarray()


to_dense = FunctionTransformer(to_dense_f, accept_sparse=True)


def logistic_regression(additional_steps, additional_params={}):
    return {
        "pipeline": Pipeline(steps=[
            *additional_steps,
            ("logistic_regression", LogisticRegression(random_state=seed_value)),
        ]),
        "params": {
            'logistic_regression__C': [0.1, 1.0, 10.0],
            'logistic_regression__solver': ['lbfgs', 'saga'],
            'logistic_regression__max_iter': [100, 200],
            'logistic_regression__penalty': ['l1', 'l2'],
            **additional_params
        }
    }


def ridge_classifier(additional_steps, additional_params={}):
    return {
        "pipeline": Pipeline(steps=[
            *additional_steps,
            ("ridge_classifier", RidgeClassifier(random_state=seed_value)),
        ]),
        "params": {
            'ridge_classifier__alpha': [0.5],
            'ridge_classifier__solver': ['sag', 'lsqr', 'sparse_cg', 'lbfgs'],
            'ridge_classifier__max_iter': [100, 200],
            **additional_params
        }
    }


def decision_tree(additional_steps, additional_params={}):
    return {
        "pipeline": Pipeline(steps=[
            *additional_steps,
            ("decision_tree", DecisionTreeClassifier(random_state=seed_value)),
        ]),
        "params": {
            'decision_tree__max_depth': [10, 30, 50],
            'decision_tree__min_samples_split': [5, 10, 30],
            'decision_tree__min_samples_leaf': [1, 5, 10],
            **additional_params
        }
    }


def random_forest(additional_steps, additional_params={}):
    return {
        "pipeline": Pipeline(steps=[
            *additional_steps,
            ("random_forest", RandomForestClassifier(random_state=seed_value)),
        ]),
        "params": {
            'random_forest__n_estimators': [150, 300],
            'random_forest__max_features': [5, 20, 50],
            'random_forest__max_depth': [20, 50],
            'random_forest__min_samples_leaf': [1, 2, 4],
            **additional_params
        }
    }


def naive_bayes(additional_steps, additional_params={}):
    return {
        "pipeline": Pipeline(steps=[
            *additional_steps,
            ("naive_bayes", GaussianNB()),
        ]),
        "params": {
            'naive_bayes__var_smoothing': [1e-9, 1e-7],
            **additional_params
        }
    }


def mlp(additional_steps, additional_params={}):
    return {
        "pipeline": Pipeline(steps=[
            *additional_steps,
            ("mlp", MLPClassifier(random_state=seed_value)),
        ]),
        "params": {
            'mlp__hidden_layer_sizes': [(50, 50)],
            'mlp__activation': ['logistic', 'relu'],
            'mlp__solver': ['adam'],
            'mlp__alpha': [0.1],
            'mlp__learning_rate': ['adaptive'],
            **additional_params
        }
    }


def svm(additional_steps, additional_params={}):
    return {
        "pipeline": Pipeline(steps=[
            *additional_steps,
            ("svm", SVC(random_state=seed_value)),
        ]),
        "params": {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['auto'],
            'svm__kernel': ['linear'],
            **additional_params
        }
    }


def get_model(hidden_layer_dim, meta):
    n_features_in_ = meta["n_features_in_"]

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(hidden_layer_dim, input_shape=(n_features_in_,), activation='relu'))
    model.add(keras.layers.Dense(hidden_layer_dim, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model


def keras_model(additional_steps, additional_params={}):
    return {
        "pipeline": Pipeline(steps=[
            *additional_steps,
            ("keras", KerasClassifier(get_model, verbose=0)),
        ]),
        "params": {
            "keras__model__hidden_layer_dim": [50, 100, 200],
            "keras__optimizer": ["adam"],
            "keras__loss": ["binary_crossentropy"],
            "keras__epochs": [1, 5, 10, 20],
            **additional_params
        }
    }


def stacking(additional_steps, additional_params={}):
    stacking_prefix = 'stacking (naive bayes, decision tree, keras) + logistic regression'

    return {
        "pipeline": Pipeline(steps=[
            *additional_steps,
            (stacking_prefix, StackingClassifier(estimators=[
                ("naive_bayes", Pipeline(steps=[("to_dense", to_dense), ("naive_bayes", GaussianNB())])),
                ("decision_tree", DecisionTreeClassifier(random_state=seed_value)),
                ("keras", KerasClassifier(get_model, verbose=0))
            ], final_estimator=LogisticRegression(random_state=seed_value)))
        ]),
        "params": {
            ('%s__naive_bayes__naive_bayes__var_smoothing' % stacking_prefix): [1e-9, 1e-7],
            ('%s__decision_tree__max_depth' % stacking_prefix): [30],
            ('%s__decision_tree__min_samples_split' % stacking_prefix): [5, 10],
            ('%s__decision_tree__min_samples_leaf' % stacking_prefix): [3, 5],
            ('%s__keras__model__hidden_layer_dim' % stacking_prefix): [100],
            ('%s__keras__optimizer' % stacking_prefix): ["adam"],
            ('%s__keras__loss' % stacking_prefix): ["binary_crossentropy"],
            ('%s__keras__epochs' % stacking_prefix): [20],
            ('%s__final_estimator__C' % stacking_prefix): [0.1],
            ('%s__final_estimator__solver' % stacking_prefix): ['saga'],
            ('%s__final_estimator__max_iter' % stacking_prefix): [200],
            ('%s__final_estimator__penalty' % stacking_prefix): ['l1'],
            **additional_params
        }
    }


preprocessor = ('preprocessor (onehot + scaler)', ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), X.select_dtypes(include=['float64', 'int']).columns),
        ('onehot', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include=['object', 'category']).columns),
    ],
    remainder='passthrough'
))

sampler = ("undersampler", RandomUnderSampler(random_state=seed_value))
pca = [("to_dense", to_dense), ("pca", PCA())]
pca_params = {'pca__n_components': [15, 22]}

tf.config.set_visible_devices([], 'GPU')

pipelines = [
    # logistic_regression([preprocessor, sampler]),
    decision_tree([preprocessor, sampler]),
    # random_forest([preprocessor, sampler]),
    naive_bayes([preprocessor, sampler, ("to_dense", to_dense)]),
    svm([preprocessor, sampler, *pca], additional_params={**pca_params}),
    # mlp([preprocessor, sampler]),
    keras_model([preprocessor, sampler]),
    stacking([preprocessor, sampler])
]
