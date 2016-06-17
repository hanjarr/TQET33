from modules import Module
from sklearn.externals import joblib
import numpy as np  

def validation():

    ''' Choose parameter file'''
    parameter_file = "/home/hannes/code/git/multi2/parameters_S1.json"

    ''' Create module object '''
    module = Module(parameter_file)

    filter_bank = np.load('/home/hannes/code/trained/multi2/S10/filter_bank.npy')
    filter_parameters = np.load('/home/hannes/code/trained/multi2/S10/filter_parameters.npy')

    print(np.shape(filter_parameters))
    
    estimator = joblib.load('/home/hannes/code/trained/multi2/S10/RegressionForest.pkl')

    error, voxel_error = module.testing(estimator, filter_bank, filter_parameters)

if __name__ == "__main__":
    validation()