import os
import pandas as pd

from sklearn.linear_model import LassoCV, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def build_model(random_state:int):

    # model parameter setting
    model_params = {

        "rf": {"model": RandomForestClassifier(random_state = random_state),
               "params": {"n_estimators": [10, 100],
                          "max_depth": [10, 20],
                          "max_features": ['auto', 'sqrt'],
                          "min_samples_leaf": [2, 4]}},

        "dt": {"model": DecisionTreeClassifier(random_state = random_state),
               "params":{"criterion": ["entropy", "gini"],
                         "max_depth": [10, 20],
                         "min_samples_leaf": [2, 4]}},

        "svc": {"model": SVC(random_state = random_state),
                "params":{"kernel": ["linear", "rbf"],
                          "C": [1, 10],
                          "gamma": [0.5, 0.01]}},

        "mlp": {"model": MLPClassifier(random_state = random_state),
                "params":{"hidden_layer_sizes": [(4,), (8,), (10,)],
                          "max_iter": [100, 200],
                          "activation": ["identity", "tanh", "relu"],
                          "solver": ["sgd", "adam"]}},

        "gbm": {"model": GradientBoostingClassifier(random_state = random_state),
                "params":{"n_estimators": [50, 100, 150],
                          "learning_rate": [0.1, 0.05],
                          "max_depth": [2, 4],
                          "min_samples_leaf": [2, 4],
                          "min_samples_split": [2, 4]}}
               }

    return model_params

