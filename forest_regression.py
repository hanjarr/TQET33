from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
import numpy as np
import time


class RegressionForest:

    def __init__(self, n_estimators, max_features, bootstrap):
        self._estimators = n_estimators
        self._max_features = max_features
        self._bootstrap = bootstrap

    def feature_selection(self, X_train, y_train, select):

        ''' Generate forest to pre select important filters '''
        estimator = self.generate_forest(X_train, y_train)
        selection_estimator = estimator['Regression forest']

        ''' Extract feature importances '''
        importances = selection_estimator.feature_importances_

        ''' Get sorted indices of most important features'''
        std = np.std([tree.feature_importances_ for tree in selection_estimator.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        ''' Return the N best filters '''
        selection_indices = indices[:select]

        # ''' Print the feature ranking'''
        # print("Feature ranking:")

        # for f in range(X_train.shape[1]):
        #   print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # ''' Plot the feature importances of the forest '''
        # plt.figure()
        # plt.title("Feature importances")
        # plt.bar(range(X_train.shape[1]), importances[indices],
        #      color="r", yerr=std[indices], align="center")
        # plt.xticks(range(X_train.shape[1]), indices)
        # plt.xlim([-1, X_train.shape[1]])
        # plt.show()

        return selection_indices

    def generate_forest(self, X_train, y_train):

        start = time.time()
        print("Train forest")

        ''' Estimators to use '''
        ESTIMATORS = {
            "Regression forest": ExtraTreesRegressor(n_estimators= self._estimators, max_features=self._max_features, 
                                                bootstrap=self._bootstrap)
        }

        trained_estimators = dict()

        for name, estimator in ESTIMATORS.items():
            trained_estimator = estimator.fit(X_train, y_train)
            trained_estimators[name] = trained_estimator

        end = time.time()
        print(end - start)

        return trained_estimators

def run_forest(estimators, X_test):

    start = time.time()
    print("Run forest")

    #regression = [estimator.predict(X_test) for name, estimator in estimators.items()]
    regression = estimators["Regression forest"].predict(X_test)

    end = time.time()
    print(end - start)

    print(np.amin(regression))
    print(regression.argmin())

    return regression

