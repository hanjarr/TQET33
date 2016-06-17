from cnn import Module
from visualization import plot_importances, plot_distribution
from sklearn.externals import joblib
import numpy as np


def main():

    ''' Choose parameter file'''
    parameter_file = "/home/hannes/code/git/cnn/parameters.json"

    ''' Create module object '''
    module = Module(parameter_file)

    ''' Forest training'''
    estimators = module.training()

    ''' Plot the filter importances '''
    #plot_importances(estimators)

    ''' Testing'''
    error, voxel_error = module.testing(estimators)

    ''' Plot the distribution '''
    plot_distribution(error, voxel_error)


if __name__ == "__main__":
    main()