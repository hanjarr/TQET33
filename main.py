from modules import Module
from utilities import plot_importances
from sklearn.externals import joblib
import numpy as np


def main():

    ''' Pre selection of filters '''
    filter_bank, filter_parameters = Module.pre_selection()

    #filter_bank = np.load('/home/hannes/code/git/filter_bank.npy')
    #filter_parameters = np.load('/home/hannes/code/git/filter_parameters.npy')


    estimators = Module.training(filter_bank, filter_parameters)

    #estimators = joblib.load('/home/hannes/code/git/RegressionForest.pkl') 

    ''' Plot the filter importances '''
    plot_importances(estimators)


    poi_error, ncc_error = Module.testing(estimators, filter_bank, filter_parameters)

    print(np.mean(poi_error))
    print(np.std(poi_error))

    print(np.mean(ncc_error))
    print(np.std(ncc_error))

if __name__ == "__main__":
    main()