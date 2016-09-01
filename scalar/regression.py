from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sknn.mlp import Regressor, Convolution, Layer

from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
import numpy as np
import time


class Regression:

    def __init__(self, n_estimators, max_features, bootstrap):
        self._estimators = n_estimators
        self._max_features = max_features
        self._bootstrap = bootstrap

    def feature_selection(self, X_train, y_train, select):

        ''' Generate forest to pre select important filters '''
        estimator = self.generate_estimator(X_train, y_train)
        selection_estimator = estimator['Regression forest']

        ''' Extract feature importances '''
        importances = selection_estimator.feature_importances_

        ''' Get sorted indices of most important features'''
        std = np.std([tree.feature_importances_ for tree in selection_estimator.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        ''' Return the N best filters '''
        selection_indices = indices[:select]

        return selection_indices


    def generate_estimator(self, X_train, y_train):

        start = time.time()
        print("Train regressor")

        ''' Estimators to use '''
        ESTIMATORS = {
            "Regression forest": ExtraTreesRegressor(n_estimators= self._estimators, 
                max_features=self._max_features, bootstrap=self._bootstrap, n_jobs = -1, oob_score=True),
            #"KNR": KNeighborsRegressor(),
            #"Linear regression": LinearRegression(),
            #"Ridge": RidgeCV(),
            #"SVR": SVR(),
            #'CNN': Regressor(layers=[Convolution("Rectifier", channels=8, kernel_shape=(3,3)), Layer("Linear")], 
            #    learning_rate=0.002, n_iter=5)
            #'NN': Regressor(layers=[Layer("Rectifier", units = 100), Layer("Rectifier", units = 30), Layer("Linear")],
            #    learning_rate=0.01, n_iter=30)
        }

        trained_estimators = dict()

        for name, estimator in ESTIMATORS.items():
            trained_estimator = estimator.fit(X_train, y_train)
            trained_estimators[name] = trained_estimator

        end = time.time()
        print(end - start)

        #print(trained_estimator.oob_score_)

        return trained_estimators

def run_estimator(estimators, X_test):

    start = time.time()
    print("Run regressor")

    regressions = dict()

    for name, estimator in estimators.items():
        regressions[name] = estimator.predict(X_test)

    end = time.time()
    print(end - start)

    return regressions


