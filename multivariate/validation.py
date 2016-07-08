from modules import Module
from sklearn.externals import joblib
import numpy as np  

def validation():

    project = 'T93'

    ''' Choose parameter file'''
    parameter_file = "/media/hannes/localDrive/trained/multivariate/"+project+"/parameters.json"

    ''' Create module object '''
    module = Module(parameter_file)

    filter_bank_z = np.load('/media/hannes/localDrive/trained/multivariate/'+project+'/filter_bank_0.npy')
    filter_bank_y = np.load('/media/hannes/localDrive/trained/multivariate/'+project+'/filter_bank_1.npy')
    filter_bank_x = np.load('/media/hannes/localDrive/trained/multivariate/'+project+'/filter_bank_2.npy')

    filter_parameter_z = np.load('/media/hannes/localDrive/trained/multivariate/'+project+'/filter_parameters_0.npy')
    filter_parameter_y = np.load('/media/hannes/localDrive/trained/multivariate/'+project+'/filter_parameters_1.npy')
    filter_parameter_x = np.load('/media/hannes/localDrive/trained/multivariate/'+project+'/filter_parameters_2.npy')

    filter_banks = [filter_bank_z, filter_bank_y, filter_bank_x]
    filter_parameters = [filter_parameter_z, filter_parameter_y, filter_parameter_x]
    
    estimator0 = joblib.load('/media/hannes/localDrive/trained/multivariate/'+project+'/RegressionForest_0.pkl')
    estimator1 = joblib.load('/media/hannes/localDrive/trained/multivariate/'+project+'/RegressionForest_1.pkl') 
    estimator2 = joblib.load('/media/hannes/localDrive/trained/multivariate/'+project+'/RegressionForest_2.pkl') 

    estimators = [estimator0, estimator1, estimator2]

    error, voxel_error = module.testing(estimators, filter_banks, filter_parameters)

if __name__ == "__main__":
    validation()