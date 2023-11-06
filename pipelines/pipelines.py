from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skrebate import ReliefF

from cursos_df import seed_value, X

preprocessor = ('preprocessor (onehot + scaler)', ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include=['object', 'category']).columns),
        ('scaler', StandardScaler(), X.select_dtypes(include=['float64', 'int']).columns)
    ],
    remainder='passthrough'
))


def to_dense_f(x):
    return x.toarray()


to_dense = FunctionTransformer(to_dense_f, accept_sparse=True)

pipelines = {
    "logistic_regression": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ("logistic_regression", LogisticRegression(random_state=seed_value)),
        ]),
        "params": {
            'logistic_regression__C': [0.1, 1.0, 10.0],
            'logistic_regression__solver': ['lbfgs', 'saga'],
            'logistic_regression__max_iter': [100, 200],
            'logistic_regression__penalty': ['l1', 'l2'],
            'logistic_regression__class_weight': ['balanced']
        }
    },
    "logistic_regression_undersample": {
        "skip": True,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("undersample", RandomUnderSampler(random_state=seed_value)),
            ("logistic_regression", LogisticRegression(random_state=seed_value)),
        ]),
        "params": {
            'logistic_regression__C': [0.1, 1.0, 10.0],
            'logistic_regression__solver': ['lbfgs', 'saga'],
            'logistic_regression__max_iter': [100, 200],
            'logistic_regression__penalty': ['l1', 'l2']
        }
    },
    "ridge_classifier": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ("ridge_classifier", RidgeClassifier(random_state=seed_value)),
        ]),
        "params": {
            'ridge_classifier__alpha': [0.5, 1.0, 2.0],
            'ridge_classifier__solver': ['sag', 'lsqr', 'sparse_cg', 'lbfgs'],
            'ridge_classifier__max_iter': [100, 200],
        }
    },
    "decision_tree": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ("decision_tree", DecisionTreeClassifier(random_state=seed_value)),
        ]),
        "params": {
            'decision_tree__max_depth': [10, 20, 30],
            'decision_tree__min_samples_split': [5, 10]
        }
    },
    "decision_tree_undersample": {
        "skip": True,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("undersample", RandomUnderSampler(random_state=seed_value)),
            ("decision_tree", DecisionTreeClassifier(random_state=seed_value)),
        ]),
        "params": {
            'decision_tree__max_depth': [10, 20, 30],
            'decision_tree__min_samples_split': [5, 10]
        }
    },
    "random_forest": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ("random_forest", RandomForestClassifier(random_state=seed_value)),
        ]),
        "params": {
            'random_forest__n_estimators': [150, 300],
            'random_forest__max_features': [5, 20, 50],
            'random_forest__max_depth': [20, 50],
            'random_forest__min_samples_leaf': [1, 2, 4]
        }
    },
    "random_forest_undersample": {
        "skip": True,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("undersample", RandomUnderSampler(random_state=seed_value)),
            ("random_forest", RandomForestClassifier(random_state=seed_value)),
        ]),
        "params": {
            'random_forest__n_estimators': [150, 300],
            'random_forest__max_depth': [10, 20],
            'random_forest__min_samples_leaf': [1, 2, 4]
        }
    },
    "naive_bayes": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ('to_dense', to_dense),
            ("naive_bayes", GaussianNB()),
        ]),
        "params": {
            'naive_bayes__var_smoothing': [1e-9, 1e-7]
        }
    },
    "bernoulli_nb": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ("bernoulli_nb", BernoulliNB()),
        ]),
        "params": {
            'bernoulli_nb__alpha': [0.5, 1.0, 2.0],
            'bernoulli_nb__binarize': [0.0, 1.0],
            'bernoulli_nb__fit_prior': [True, False],
        }
    },
    "svm": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ("svm", SVC(random_state=seed_value)),
        ]),
        "params": {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['scale', 'auto'],
            'svm__kernel': ['linear', 'rbf', 'poly']
        }
    },
    "svm_pca": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ('to_dense', to_dense),
            ('pca', PCA()),
            ("svm", SVC(random_state=seed_value)),
        ]),
        "params": {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['scale', 'auto'],
            'svm__kernel': ['linear', 'rbf', 'poly'],
            'pca__n_components': [10, 15, 20, 22]
        }
    },
    "svm_relief": {
        "skip": True,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ('to_dense', to_dense),
            ('relief', ReliefF()),
            ("svm", SVC(random_state=seed_value)),
        ]),
        "params": {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['scale', 'auto'],
            'svm__kernel': ['linear', 'rbf'],
            'relief__n_neighbors': [10, 20, 30],
            'relief__n_features_to_select': [5, 10, 15]
        }
    },
    "knn_pca": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ('to_dense', to_dense),
            ('pca', PCA()),
            ("knn", KNeighborsClassifier()),
        ]),
        "params": {
            'knn__n_neighbors': [3, 5, 7],
            'knn__weights': ['uniform', 'distance'],
            'knn__algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'pca__n_components': [10, 15, 20, 22]
        }
    },
    "mlp": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ("mlp", MLPClassifier(random_state=seed_value)),
        ]),
        "params": {
            'mlp__hidden_layer_sizes': [(50, 50), (100,)],
            'mlp__activation': ['tanh', 'relu'],
            'mlp__solver': ['sgd', 'adam'],
            'mlp__alpha': [0.0001, 0.05],
            'mlp__learning_rate': ['constant', 'adaptive'],
        }
    }
}
