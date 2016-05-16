from modules import Module
from utilities import plot_importances, plot_distribution
from sklearn.externals import joblib
import numpy as np


def main():

    ''' Choose parameter file'''
    #parameter_file = "/home/hannes/code/git/multivariate/parameters.json"

    ''' Create module object '''
    #module = Module(parameter_file)

    ''' Pre selection of filters '''
    #filter_banks, filter_parameters = module.pre_selection()

    #filter_bank = np.load('/home/hannes/code/trained/LeftFemur/test23/filter_bank.npy')
    #filter_parameters = np.load('/home/hannes/code/trained/LeftFemur/test23/filter_parameters.npy')

    #estimators = module.training(filter_banks, filter_parameters)

    #estimator0 = joblib.load('/home/hannes/code/git/multivariate/RegressionForest_0.pkl')
    #estimator1 = joblib.load('/home/hannes/code/git/multivariate/RegressionForest_1.pkl') 
    #estimator2 = joblib.load('/home/hannes/code/git/multivariate/RegressionForest_2.pkl') 

    #estimators = [estimator0, estimator1, estimator2]
 

    ''' Plot the filter importances '''
    #plot_importances(estimators)

    #error, voxel_error = module.testing(estimators, filter_banks, filter_parameters)

    error = np.load('/home/hannes/code/trained/multivariate/T91/error.npy')
    voxel_error = np.load('/home/hannes/code/trained/multivariate/T91/voxel_error.npy')

    plot_distribution(error, voxel_error)


if __name__ == "__main__":
    main()