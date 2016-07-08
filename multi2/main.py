from modules import Module
from visualization import plot_importances, plot_distribution
from sklearn.externals import joblib
import numpy as np


def main():

    ''' Choose parameter file'''
    parameter_file = "/home/hannes/code/git/multi2/parameters/parameters_T11.json"

    ''' Create module object '''
    module = Module(parameter_file)

    ''' Pre selection of filters '''
    filter_bank, filter_parameters = module.pre_selection()

    #filter_bank = np.load('/home/hannes/code/git/multi2/filter_bank.npy')
    #filter_parameters = np.load('/home/hannes/code/git/multi2/filter_parameters.npy')

    ''' Forest training'''
    estimators = module.training(filter_bank, filter_parameters)

    #estimators = joblib.load('/home/hannes/code/git/multi2/forest/RegressionForest.pkl')

    ''' Testing'''
    #error, voxel_error = module.testing(estimators, filter_bank, filter_parameters)

    ''' Plot the distribution '''
    #plot_distribution(error, voxel_error)


if __name__ == "__main__":
    main()