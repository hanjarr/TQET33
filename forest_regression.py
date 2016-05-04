from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
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

        return selection_indices

    def feature_selection_2(self, X, y):
        scores = defaultdict(list)
 
        #crossvalidate the scores on a number of different random splits of the data
        for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            estimator = self.generate_forest(X_train, y_train)
            acc = r2_score(Y_test, rf.predict(X_test))

            for i in range(X.shape[1]):
                X_t = X_test.copy()
                np.random.shuffle(X_t[:, i])
                shuff_acc = r2_score(Y_test, rf.predict(X_t))
                scores[names[i]].append((acc-shuff_acc)/acc)

            print "Features sorted by their score:"
            print sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores.items()], reverse=True)

    def generate_forest(self, X_train, y_train):

        start = time.time()
        print("Train forest")

        ''' Estimators to use '''
        ESTIMATORS = {
            "Regression forest": ExtraTreesRegressor(n_estimators= self._estimators, 
                max_features=self._max_features, bootstrap=self._bootstrap, n_jobs = -1, oob_score=True)
        }

        trained_estimators = dict()

        for name, estimator in ESTIMATORS.items():
            trained_estimator = estimator.fit(X_train, y_train)
            trained_estimators[name] = trained_estimator

        end = time.time()
        print(end - start)

        print(trained_estimator.oob_score_)

        return trained_estimators

def run_forest(estimators, X_test):

    start = time.time()
    print("Run forest")

    #regression = [estimator.predict(X_test) for name, estimator in estimators.items()]
    regression = estimators["Regression forest"].predict(X_test)

    end = time.time()
    print(end - start)

    return regression


