from modules import Module
from sklearn.externals import joblib
import numpy as np  

def validation():

    project = 'RF1'

    ''' Choose parameter file'''
    parameter_file = '/home/hannes/code/trained/multi2/' + project + '/parameters.json'

    ''' Create module object '''
    module = Module(parameter_file)

    filter_bank = np.load('/home/hannes/code/trained/multi2/' + project + '/filter_bank.npy')
    filter_parameters = np.load('/home/hannes/code/trained/multi2/' + project + '/filter_parameters.npy')
    
    estimator = joblib.load('/home/hannes/code/trained/multi2/' + project + '/RegressionForest.pkl')

    error, voxel_error = module.testing(estimator, filter_bank, filter_parameters)

if __name__ == "__main__":
    validation()