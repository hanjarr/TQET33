''' Feature extraction '''
import numpy as np
from scipy.ndimage.filters import convolve as conv
from scipy.ndimage.filters import sobel
import matplotlib.pyplot as plt
import random as rand
import time


class Feature:

    def __init__(self, nbr_of_filters, patch_size):
        self._nbr_of_filters = nbr_of_filters
        self._patch_size = patch_size

    def feature_extraction(self, water_target, fat_target, filter_bank, filter_parameters):

        #start = time.time()
        #print("Feature extraction")

        #filter_bank, parameters = self.generate_haar()
        #param = list(filter_parameters)

        ''' Allocate memory for storing feature vectors ''' 
        water_features = np.zeros((len(np.ravel(water_target)),filter_bank.shape[0]))
        fat_features = water_features.copy()


        # if filter_type == 'Sobel':
        #     for direction in range(2,-1,-1):
        #         derivative = sobel(target, direction)
        #         extracted_features.append(derivative)

        # else:
        for ind, haar_filter in enumerate(filter_bank):
            rep_filter = np.tile(haar_filter[None,:,:], ((filter_parameters[ind][0]),1,1))

            ''' Zero pad the Haar convolutional kernel '''
            front_zeros = np.zeros((filter_parameters[ind][1], self._patch_size, self._patch_size))
            back_zeros = np.zeros((filter_parameters[ind][2], self._patch_size, self._patch_size))

            complete_filter = np.vstack((np.vstack((front_zeros,rep_filter)),back_zeros))

            ''' Convolve the water and fat signal with the filter bank'''
            water_conv = conv(water_target, complete_filter, mode='constant', cval=0.0)
            #fat_conv = conv(fat_target, complete_filter, mode='constant', cval=0.0)

            water_features[:,ind] = np.ravel(water_conv)
            #fat_features[:,ind] = np.ravel(fat_conv)


        #end = time.time()
        #print(end - start)

        return water_features, fat_features


    def generate_haar(self):
        haar_bank = np.zeros([self._nbr_of_filters, self._patch_size, self._patch_size])
        z_size, front_zeros, back_zeros = [], [], []

        ''' Generate 2D-filters and 3D parameters '''
        for filt in range(0, self._nbr_of_filters):
            haar_size = [rand.randint(1, self._patch_size) for _ in range(0,3)]
            origin = [rand.randint(0, self._patch_size-haar_size[1]), rand.randint(0, self._patch_size-haar_size[2])]
            haar_bank[filt,origin[0]:origin[0] + haar_size[1], origin[1]:origin[1] + haar_size[2]] = 1/np.prod(haar_size)

            z_size.append(haar_size[0])
            front_zeros.append(rand.randint(0, self._patch_size-z_size[filt]))
            back_zeros.append(self._patch_size-(front_zeros[filt] + z_size[filt]))

        ''' Zip filter parameters '''
        haar_parameters = list(zip(*[z_size, front_zeros, back_zeros]))

        return haar_bank, haar_parameters

