''' Feature extraction '''
import numpy as np
from scipy.ndimage.filters import convolve as conv
from scipy.ndimage.filters import sobel
import matplotlib.pyplot as plt
import random as rand


class Feature:

    def __init__(self, nbr_of_filters, patch_size):
        self._nbr_of_filters = nbr_of_filters
        self._patch_size = patch_size

    def feature_extraction(self, target, filter_type = 'Haar'):
        filter_bank, parameters = self.generate_haar()
        param = list(parameters)

        ''' Allocate memory for storing feature vectors ''' 
        extracted_features = np.zeros((len(np.ravel(target)),filter_bank.shape[0]))


        if filter_type == 'Sobel':
            for direction in range(2,-1,-1):
                derivative = sobel(target, direction)
                extracted_features.append(derivative)

        else:
            for ind, haar_filter in enumerate(filter_bank):
                rep_filter = np.tile(haar_filter[None,:,:], ((param[ind][0]),1,1))

                ''' Zero pad the Haar convolutional kernel '''
                front_zeros = np.zeros((param[ind][1], self._patch_size, self._patch_size))
                back_zeros = np.zeros((param[ind][2], self._patch_size, self._patch_size))

                complete_filter = np.vstack((np.vstack((front_zeros,rep_filter)),back_zeros))

                target_conv = conv(target, complete_filter, mode='constant', cval=0.0)
                extracted_features[:,ind] = np.ravel(target_conv)

        return extracted_features


    def generate_haar(self):
        haar_bank = np.zeros([self._nbr_of_filters, self._patch_size, self._patch_size])
        z_size, front_zeros, back_zeros = [], [], []

        ''' Generate 2D-filters and 3D parameters '''
        for filt in range(0, self._nbr_of_filters):
            haar_size = [rand.randint(1, self._patch_size) for _ in range(0,3)]
            origin = [rand.randint(0, self._patch_size-haar_size[1]), rand.randint(0, self._patch_size-haar_size[2])]
            haar_bank[filt,origin[0]:origin[0] + haar_size[1], origin[1]:origin[1] + haar_size[2]] = 1

            z_size.append(haar_size[0])
            front_zeros.append(rand.randint(0, self._patch_size-z_size[filt]))
            back_zeros.append(self._patch_size-(front_zeros[filt] + z_size[filt]))

        ''' Zip filter parameters '''
        haar_parameters = zip(*[z_size, front_zeros, back_zeros])

        return haar_bank, haar_parameters

