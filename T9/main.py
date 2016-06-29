from modules import Module
from visualization import plot_importances, plot_distribution
from sklearn.externals import joblib
import numpy as np


def main():

    ''' Choose parameter file'''
    parameter_file = '/media/hannes/localDrive/trained/multi2/T9/parameters/parameters.json'

    ''' Create module object '''
    module = Module(parameter_file)

    ''' Testing'''
    error, voxel_error = module.testing()


if __name__ == "__main__":
    main()