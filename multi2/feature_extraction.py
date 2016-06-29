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

    def haar_extraction(self, water_target, fat_target, filter_bank, filter_parameters):

        start = time.time()
        print("Feature extraction")

        ''' Allocate memory for storing feature vectors ''' 
        water_features = np.zeros((len(np.ravel(water_target)), self._nbr_of_filters))
        fat_features = water_features.copy()


        for ind, haar_filter in enumerate(filter_bank):

            first_cubical = haar_filter[0]
            second_cubical = haar_filter[1]

            rep_first = np.tile(first_cubical[None,:,:], ((filter_parameters[ind][0][0]),1,1))
            rep_second = np.tile(second_cubical[None,:,:], ((filter_parameters[ind][1][0]),1,1))

            ''' Zero pad the Haar convolutional kernel '''
            front_zeros = np.zeros((filter_parameters[ind][0][1], self._patch_size, self._patch_size))
            back_zeros = np.zeros((filter_parameters[ind][0][2], self._patch_size, self._patch_size))

            front_zeros_ = np.zeros((filter_parameters[ind][1][1], self._patch_size, self._patch_size))
            back_zeros_ = np.zeros((filter_parameters[ind][1][2], self._patch_size, self._patch_size))

            first_filter = np.vstack((np.vstack((front_zeros, rep_first)), back_zeros))
            second_filter = np.vstack((np.vstack((front_zeros_, rep_second)), back_zeros_))

            complete_filter = first_filter + second_filter


            ''' Convolve the water and fat signal with the filter bank'''
            water_conv = conv(water_target, complete_filter, mode='constant', cval=0.0)
            fat_conv = conv(fat_target, complete_filter, mode='constant', cval=0.0)

            water_features[:,ind] = np.ravel(water_conv)
            fat_features[:,ind] = np.ravel(fat_conv)


        end = time.time()
        print(end - start)

        return water_features, fat_features

    def sobel_extraction(self, water_target, fat_target):

        water_features = np.zeros((water_target.shape))
        fat_features = water_features.copy()

        for direction in range(2,-1,-1):
            deriv = sobel(water_target, direction)
            water_features = water_features + deriv**2
        water_features = np.sqrt(water_features)
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

    def generate_haar_(self):
        haar_bank = np.zeros([self._nbr_of_filters, self._patch_size, self._patch_size])
        haar_bank_ = haar_bank.copy()

        z_size, front_zeros, back_zeros = [], [], []
        z_size_, front_zeros_, back_zeros_ = [], [], []

        max_haar_size = self._patch_size
        min_haar_size = 2


        ''' Generate 2D-filters and 3D parameters '''
        for filt in range(0, self._nbr_of_filters):
    
            ''' Randomize parameters for first cubical region '''
            haar_size = [rand.randint(min_haar_size, max_haar_size) for _ in range(0,3)]
            origin = [rand.randint(0, max_haar_size - haar_size[1]), rand.randint(0, max_haar_size - haar_size[2])]
            haar_bank[filt,origin[0]:origin[0] + haar_size[1], origin[1]:origin[1] + haar_size[2]] = 1/np.prod(haar_size)

            z_size.append(haar_size[0])
            front_zeros.append(rand.randint(0, self._patch_size-z_size[filt]))
            back_zeros.append(self._patch_size-(front_zeros[filt] + z_size[filt]))

            ''' Randomize parameters for second cubical region '''
            rand.shuffle(haar_size)
            if rand.random() > 0.5:
                origin = [rand.randint(0, max_haar_size - haar_size[1]), rand.randint(0, max_haar_size - haar_size[2])]
                haar_bank_[filt,origin[0]:origin[0] + haar_size[1], origin[1]:origin[1] + haar_size[2]] = -1/np.prod(haar_size)

            z_size_.append(haar_size[0])
            front_zeros_.append(rand.randint(0, self._patch_size-z_size_[filt]))
            back_zeros_.append(self._patch_size-(front_zeros_[filt] + z_size_[filt]))

        ''' Zip filter parameters '''
        first_haar = list(zip(*[z_size, front_zeros, back_zeros]))
        second_haar = list(zip(*[z_size_, front_zeros_, back_zeros_]))

        haar_parameters = list(zip(first_haar, second_haar))

        filter_bank = list(zip(haar_bank, haar_bank_))

        return filter_bank, haar_parameters

