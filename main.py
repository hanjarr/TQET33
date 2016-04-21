from modules import Module
from utilities import plot_importances, plot_distribution
from sklearn.externals import joblib
import numpy as np


def main():

    ''' Pre selection of filters '''
    #filter_bank, filter_parameters = Module.pre_selection()

    #filter_bank = np.load('/home/hannes/code/trained/RightFemur/test2/filter_bank.npy')
    #filter_parameters = np.load('/home/hannes/code/trained/RightFemur/test2/filter_parameters.npy')

    #estimators = Module.training(filter_bank, filter_parameters)

    #estimators = joblib.load('/home/hannes/code/trained/RightFemur/test2/RegressionForest.pkl') 

    ''' Plot the filter importances '''
    #plot_importances(estimators)


    #error, voxel_error = Module.testing(estimators, filter_bank, filter_parameters)


    error = np.load('/home/hannes/code/trained/LeftFemur/test23/error.npy')
    voxel_error = np.load('/home/hannes/code/trained/LeftFemur/test23/voxel_error.npy')

    plot_distribution(error, voxel_error)


if __name__ == "__main__":
    main()