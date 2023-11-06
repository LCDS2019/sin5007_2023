import numpy as np
import random

from deap import base, creator, tools, algorithms
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score

class GeneticFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_gen=10, size=50, cxpb=0.5, mutpb=0.2, random_state=None):
        # Parâmetros do GA
        self.n_gen = n_gen  # Número de gerações
        self.size = size  # Tamanho da população
        self.cxpb = cxpb  # Probabilidade de crossover
        self.mutpb = mutpb  # Probabilidade de mutação

        # Estimador
        self.estimator = estimator

        # DEAP setup
        if random_state:
            random.seed(random_state)
            np.random.seed(random_state)

        # Criação do tipo Fitness e Individual
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Operadores do GA
        self.toolbox.register("evaluate", self.evalFeatures)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selBest)

    def evalFeatures(self, individual):
        # Reduzir o conjunto de dados apenas às características selecionadas
        features = [i for i, bit in enumerate(individual) if bit == 1]
        if len(features) == 0:
            return 0,

        # Avaliar com o estimador fornecido
        return (np.mean(cross_val_score(self.estimator, self._X[:, features], self._y, cv=5)),)

    def fit(self, X, y):
        # Guardar dados para uso no método de avaliação
        self._X, self._y = X, y
        self.n_features_ = X.shape[1]

        # Atualizar o comprimento do indivíduo baseado no número de características
        self.toolbox.unregister("individual")
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool,
                              n=self.n_features_)

        # Executar o GA
        pop = self.toolbox.population(n=self.size)
        hof = tools.HallOfFame(1)
        algorithms.eaSimple(pop, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.n_gen,
                            halloffame=hof, verbose=False)

        # Guardar o melhor conjunto de características
        self.best_individual_ = hof[0]
        self.support_ = [bool(x) for x in self.best_individual_]

        return self

    def transform(self, X):
        # Reduzir para as características selecionadas
        return X[:, self.support_]

    def fit_transform(self, X, y):
        # Combinação do fit e transform em um único passo
        return self.fit(X, y).transform(X)
