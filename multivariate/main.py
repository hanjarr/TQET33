from modules import Module
from visualization import plot_importances, plot_distribution
from sklearn.externals import joblib
import numpy as np


def main():

    ''' Choose parameter file'''
    parameter_file = "/home/hannes/code/git/multivariate/parameters_S1.json"

    ''' Create module object '''
    module = Module(parameter_file)

    ''' Pre selection of filters '''
    filter_banks, filter_parameters = module.pre_selection()

    ''' Forest training'''
    estimators = module.training(filter_banks, filter_parameters)

    ''' Plot the filter importances '''
    #plot_importances(estimators)

    ''' Testing'''
    error, voxel_error = module.testing(estimators, filter_banks, filter_parameters)

    ''' Plot the distribution '''
    plot_distribution(error, voxel_error)


if __name__ == "__main__":
    main()