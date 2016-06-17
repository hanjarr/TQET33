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

        self.forest = ExtraTreesRegressor(n_estimators= n_estimators, 
            max_features=max_features, 
            bootstrap=bootstrap, 
            n_jobs = 6, 
            oob_score=True, 
            warm_start = True)

    def feature_selection(self, select):

        ''' Extract feature importances '''
        importances = self.forest.feature_importances_

        ''' Get sorted indices of most important features'''
        std = np.std([tree.feature_importances_ for tree in self.forest.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        ''' Return the N best filters '''
        selection_indices = indices[:select]

        return selection_indices


    def generate_estimator(self, X_train, y_train):

        start = time.time()
        print("Train regressor")

        self.forest = self.forest.fit(X_train, y_train)
        end = time.time()
        print(end - start)

        return None

def run_estimator(regressor, X_test):

    start = time.time()
    print("Run regressor")

    regression = regressor.forest.predict(X_test)

    end = time.time()
    print(end - start)

    return regression


