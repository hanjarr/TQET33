from sklearn.ensemble import ExtraTreesRegressor

class RegressionForest:

    def __init__(self, n_estimators, max_features, bootstrap):
        self._estimators = n_estimators
        self._max_features = max_features
        self._bootstrap = bootstrap

    def generateForest(self, X_train, y_train):

        ''' Estimators to use '''
        ESTIMATORS = {
            "Extra trees": ExtraTreesRegressor(n_estimators= self._estimators, max_features=self._max_features, 
                                                bootstrap=self._bootstrap)
        }

        for name, estimator in ESTIMATORS.items():
            estimator.fit(X_train, y_train)
            importances = estimator.feature_importances_

        std = np.std([tree.feature_importances_ for tree in estimator.estimators_],
             axis=0)
        indices = np.argsort(importances)[::-1]

        ''' Print the feature ranking'''
        print("Feature ranking:")

        for f in range(X_train.shape[1]):
           print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices],
              color="r", yerr=std[indices], align="center")
        plt.xticks(range(X_train.shape[1]), indices)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()