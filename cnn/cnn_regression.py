from sknn.mlp import Regressor, Layer, Convolution
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
import numpy as np
import time


class NeuralNetwork:

    def __init__(self, kernel_shape, iterations, learning_rate):
        self._iter = iterations
        self._learning_rate = learning_rate
        self._kernel_shape = kernel_shape


    def generate_cnn(self, X_train, y_train):

        start = time.time()
        print("Train network")

        ''' Estimators to use '''
        ESTIMATORS = {
            "CNN": Regressor(layers=[Convolution("Rectifier", channels=12, kernel_shape=(self._kernel_shape, self._kernel_shape)), 
                Convolution("Rectifier", channels=8, kernel_shape=(self._kernel_shape, self._kernel_shape)), Layer("Linear")], 
                learning_rate=self._learning_rate, n_iter=self._iter)
        }

        trained_estimators = dict()

        for name, estimator in ESTIMATORS.items():
            trained_estimator = estimator.fit(X_train, y_train)
            trained_estimators[name] = trained_estimator

        end = time.time()
        print(end - start)

        return trained_estimators


def run_cnn(estimators, X_test):

    start = time.time()
    print("Run forest")

    #regression = [estimator.predict(X_test) for name, estimator in estimators.items()]
    regression = estimators["CNN"].predict(X_test)

    end = time.time()
    print(end - start)

    return regression


