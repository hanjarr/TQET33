from modules import Module
from sklearn.externals import joblib
import numpy as np  

def validation():

    project = 'test7'

    filter_bank = np.load('/media/hannes/localDrive/trained/scalar/RightFemur/' + project + '/filter_bank.npy')
    filter_parameters = np.load('/media/hannes/localDrive/trained/scalar/RightFemur/' + project + '/filter_parameters.npy')
    
    estimator = joblib.load('/media/hannes/localDrive/trained/scalar/RightFemur/' + project + '/RegressionForest.pkl')

    error, voxel_error = Module.testing(estimator, filter_bank, filter_parameters)

if __name__ == "__main__":
    validation()