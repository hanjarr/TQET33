from modules import Module
from sklearn.externals import joblib
import numpy as np  

def validation():

    project = 'LeftFemur/test21'

    filter_bank = np.load('/media/hannes/localDrive/trained/scalar/' + project + '/filter_bank.npy')
    filter_parameters = np.load('/media/hannes/localDrive/trained/scalar/' + project + '/filter_parameters.npy')
    
    estimator = joblib.load('/media/hannes/localDrive/trained/scalar/' + project + '/RegressionForest.pkl')

    error, voxel_error = Module.testing(estimator, filter_bank, filter_parameters)

if __name__ == "__main__":
    validation()