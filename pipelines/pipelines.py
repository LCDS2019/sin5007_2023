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
        ('scaler', StandardScaler(), X.select_dtypes(include=['float64', 'int']).columns),
        ('onehot', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include=['object', 'category']).columns),
    ],
    remainder='passthrough'
))


def to_dense_f(x):
    return x.toarray()


to_dense = FunctionTransformer(to_dense_f, accept_sparse=True)

undersample_pipelines = {
    "logistic_regression_undersample": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("undersampler", RandomUnderSampler(random_state=seed_value)),
            ("logistic_regression", LogisticRegression(random_state=seed_value)),
        ]),
        "params": {
            'logistic_regression__C': [0.1, 1.0, 10.0],
            'logistic_regression__solver': ['lbfgs', 'saga'],
            'logistic_regression__max_iter': [100, 200],
            'logistic_regression__penalty': ['l1', 'l2']
        }
    },
    "ridge_classifier_undersample": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("undersampler", RandomUnderSampler(random_state=seed_value)),
            ("ridge_classifier", RidgeClassifier(random_state=seed_value)),
        ]),
        "params": {
            'ridge_classifier__alpha': [0.5],
            'ridge_classifier__solver': ['sag', 'lsqr', 'sparse_cg', 'lbfgs'],
            'ridge_classifier__max_iter': [100, 200],
        }
    },
    "decision_tree_undersample": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("undersampler", RandomUnderSampler(random_state=seed_value)),
            ("decision_tree", DecisionTreeClassifier(random_state=seed_value)),
        ]),
        "params": {
            'decision_tree__max_depth': [10, 30],
            'decision_tree__min_samples_split': [5, 10]
        }
    },
    "random_forest_undersample": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("undersampler", RandomUnderSampler(random_state=seed_value)),
            ("random_forest", RandomForestClassifier(random_state=seed_value)),
        ]),
        "params": {
            'random_forest__n_estimators': [150, 300],
            'random_forest__max_features': [5, 20, 50],
            'random_forest__max_depth': [20, 50],
            'random_forest__min_samples_leaf': [1, 2, 4]
        }
    },
    "naive_bayes_undersample": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("undersampler", RandomUnderSampler(random_state=seed_value)),
            ('to_dense', to_dense),
            ("naive_bayes", GaussianNB()),
        ]),
        "params": {
            'naive_bayes__var_smoothing': [1e-9, 1e-7]
        }
    },
    "bernoulli_nb_undersample": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("undersampler", RandomUnderSampler(random_state=seed_value)),
            ("bernoulli_nb", BernoulliNB()),
        ]),
        "params": {
            'bernoulli_nb__alpha': [0.5, 1.0, 2.0],
            'bernoulli_nb__binarize': [0.0, 1.0],
            'bernoulli_nb__fit_prior': [True, False],
        }
    },
    "svm_pca_undersample": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("undersampler", RandomUnderSampler(random_state=seed_value)),
            ('to_dense', to_dense),
            ('pca', PCA()),
            ("svm", SVC(random_state=seed_value)),
        ]),
        "params": {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['auto'],
            'svm__kernel': ['linear'],
            'pca__n_components': [15, 22]
        }
    },
    "svm_undersample": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("undersampler", RandomUnderSampler(random_state=seed_value)),
            ("svm", SVC(random_state=seed_value)),
        ]),
        "params": {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['auto'],
            'svm__kernel': ['linear']
        }
    },
    "svm_relief_undersample": {
        "skip": True,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("undersampler", RandomUnderSampler(random_state=seed_value)),
            ('to_dense', to_dense),
            ('relief', ReliefF()),
            ("svm", SVC(random_state=seed_value)),
        ]),
        "params": {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['auto'],
            'svm__kernel': ['linear', 'rbf'],
            'relief__n_neighbors': [10, 20, 30],
            'relief__n_features_to_select': [5, 10, 15]
        }
    },
    "knn_pca_undersample": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("undersampler", RandomUnderSampler(random_state=seed_value)),
            ('to_dense', to_dense),
            ('pca', PCA()),
            ("knn", KNeighborsClassifier()),
        ]),
        "params": {
            'knn__n_neighbors': [3, 5],
            'knn__weights': ['uniform', 'distance'],
            'knn__algorithm': ['auto'],
            'pca__n_components': [15, 22]
        }
    },
    "mlp_undersample": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("undersampler", RandomUnderSampler(random_state=seed_value)),
            ("mlp", MLPClassifier(random_state=seed_value)),
        ]),
        "params": {
            'mlp__hidden_layer_sizes': [(50, 50)],
            'mlp__activation': ['logistic', 'relu'],
            'mlp__solver': ['adam'],
            'mlp__alpha': [0.1],
            'mlp__learning_rate': ['adaptive'],
        }
    },
}

oversample_pipelines = {
    "logistic_regression_oversample": {
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
    "ridge_classifier_oversample": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ("ridge_classifier", RidgeClassifier(random_state=seed_value)),
        ]),
        "params": {
            'ridge_classifier__alpha': [0.5],
            'ridge_classifier__solver': ['sag', 'lsqr', 'sparse_cg', 'lbfgs'],
            'ridge_classifier__max_iter': [100, 200],
        }
    },
    "decision_tree_oversample": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ("decision_tree", DecisionTreeClassifier(random_state=seed_value)),
        ]),
        "params": {
            'decision_tree__max_depth': [10, 30],
            'decision_tree__min_samples_split': [5, 10]
        }
    },
    "random_forest_oversample": {
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
    "naive_bayes_oversample": {
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
    "bernoulli_nb_oversample": {
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
    "svm_oversample": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ("svm", SVC(random_state=seed_value)),
        ]),
        "params": {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['auto'],
            'svm__kernel': ['linear']
        }
    },
    "svm_pca_oversample": {
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
            'svm__gamma': ['auto'],
            'svm__kernel': ['linear'],
            'pca__n_components': [15, 22]
        }
    },
    "svm_relief_oversample": {
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
            'svm__gamma': ['auto'],
            'svm__kernel': ['linear', 'rbf'],
            'relief__n_neighbors': [10, 20, 30],
            'relief__n_features_to_select': [5, 10, 15]
        }
    },
    "knn_pca_oversample": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ('to_dense', to_dense),
            ('pca', PCA()),
            ("knn", KNeighborsClassifier()),
        ]),
        "params": {
            'knn__n_neighbors': [3, 5],
            'knn__weights': ['uniform', 'distance'],
            'knn__algorithm': ['auto'],
            'pca__n_components': [15, 22]
        }
    },
    "mlp_oversample": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("oversampler", RandomOverSampler(random_state=seed_value)),
            ("mlp", MLPClassifier(random_state=seed_value)),
        ]),
        "params": {
            'mlp__hidden_layer_sizes': [(50, 50)],
            'mlp__activation': ['logistic', 'relu'],
            'mlp__solver': ['adam'],
            'mlp__alpha': [0.1],
            'mlp__learning_rate': ['adaptive'],
        }
    }
}

pipelines = {
    "logistic_regression": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
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
    "ridge_classifier": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("ridge_classifier", RidgeClassifier(random_state=seed_value)),
        ]),
        "params": {
            'ridge_classifier__alpha': [0.5],
            'ridge_classifier__solver': ['sag', 'lsqr', 'sparse_cg', 'lbfgs'],
            'ridge_classifier__max_iter': [100, 200],
        }
    },
    "decision_tree": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("decision_tree", DecisionTreeClassifier(random_state=seed_value)),
        ]),
        "params": {
            'decision_tree__max_depth': [10, 30],
            'decision_tree__min_samples_split': [5, 10]
        }
    },
    "random_forest": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("random_forest", RandomForestClassifier(random_state=seed_value)),
        ]),
        "params": {
            'random_forest__n_estimators': [150, 300],
            'random_forest__max_features': [5, 20, 50],
            'random_forest__max_depth': [20, 50],
            'random_forest__min_samples_leaf': [1, 2, 4]
        }
    },
    "naive_bayes": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
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
            ("svm", SVC(random_state=seed_value)),
        ]),
        "params": {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['auto'],
            'svm__kernel': ['linear']
        }
    },
    "svm_pca": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ('to_dense', to_dense),
            ('pca', PCA()),
            ("svm", SVC(random_state=seed_value)),
        ]),
        "params": {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['auto'],
            'svm__kernel': ['linear'],
            'pca__n_components': [15, 22]
        }
    },
    "svm_relief": {
        "skip": True,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ('to_dense', to_dense),
            ('relief', ReliefF()),
            ("svm", SVC(random_state=seed_value)),
        ]),
        "params": {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['auto'],
            'svm__kernel': ['linear', 'rbf'],
            'relief__n_neighbors': [10, 20, 30],
            'relief__n_features_to_select': [5, 10, 15]
        }
    },
    "knn_pca": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ('to_dense', to_dense),
            ('pca', PCA()),
            ("knn", KNeighborsClassifier()),
        ]),
        "params": {
            'knn__n_neighbors': [3, 5],
            'knn__weights': ['uniform', 'distance'],
            'knn__algorithm': ['auto'],
            'pca__n_components': [15, 22]
        }
    },
    "mlp": {
        "skip": False,
        "pipeline": Pipeline(steps=[
            preprocessor,
            ("mlp", MLPClassifier(random_state=seed_value)),
        ]),
        "params": {
            'mlp__hidden_layer_sizes': [(50, 50)],
            'mlp__activation': ['logistic', 'relu'],
            'mlp__solver': ['adam'],
            'mlp__alpha': [0.1],
            'mlp__learning_rate': ['adaptive'],
        }
    }
}

all_pipelines = {**pipelines, **undersample_pipelines, **oversample_pipelines}
