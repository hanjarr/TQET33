from modules import Module
from utilities import plot_importances, plot_distribution
from sklearn.externals import joblib
import numpy as np
import glob, os



def main():

    ''' Choose parameter file'''
    #parameter_file = os.getcwd() + '/' + str(glob.glob("*.json")[0])

    parameter_file = "/home/hannes/code/git/multivariate/T9/parameters.json"

    ''' Create module object '''
    module = Module(parameter_file)

    ''' Pre selection of filters '''
    #filter_banks, filter_parameters = module.pre_selection()

    filter_bank_z = np.load('/home/hannes/code/trained/multivariate/T91/filter_bank_0.npy')
    filter_bank_y = np.load('/home/hannes/code/trained/multivariate/T91/filter_bank_1.npy')
    filter_bank_x = np.load('/home/hannes/code/trained/multivariate/T91/filter_bank_2.npy')

    filter_parameter_z = np.load('/home/hannes/code/trained/multivariate/T91/filter_parameters_0.npy')
    filter_parameter_y = np.load('/home/hannes/code/trained/multivariate/T91/filter_parameters_1.npy')
    filter_parameter_x = np.load('/home/hannes/code/trained/multivariate/T91/filter_parameters_2.npy')

    filter_banks = [filter_bank_z, filter_bank_y, filter_bank_x]
    filter_parameters = [filter_parameter_z, filter_parameter_y, filter_parameter_x]


    estimators = module.training(filter_banks, filter_parameters)

    #estimator0 = joblib.load('/home/hannes/code/git/multivariate/RegressionForest_0.pkl')
    #estimator1 = joblib.load('/home/hannes/code/git/multivariate/RegressionForest_1.pkl') 
    #estimator2 = joblib.load('/home/hannes/code/git/multivariate/RegressionForest_2.pkl') 

    #estimators = [estimator0, estimator1, estimator2]
 

    ''' Plot the filter importances '''
    #plot_importances(estimators)

    error, voxel_error = module.testing(estimators, filter_banks, filter_parameters)

    #error = np.load('/home/hannes/code/trained/multivariate/RF1/error.npy')
    #voxel_error = np.load('/home/hannes/code/trained/multivariate/RF1/voxel_error.npy')

    plot_distribution(error, voxel_error)


if __name__ == "__main__":
    main()