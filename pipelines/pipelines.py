from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
            'decision_tree__max_depth': [10, 30],
            'decision_tree__min_samples_split': [5, 10],
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

pipelines = [
    logistic_regression([preprocessor, sampler]),
    decision_tree([preprocessor, sampler]),
    random_forest([preprocessor, sampler]),
    naive_bayes([preprocessor, sampler, ("to_dense", to_dense)]),
    svm([preprocessor, sampler, *pca], additional_params={**pca_params}),
    mlp([preprocessor, sampler])
]
