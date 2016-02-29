''' Feature extraction '''
import numpy as np
import random as rand

def feature_extraction(target, imgSize, filter_bank = 'Haar'):


def generate_haar(nbr_filters, patch_size):
    haar_bank = np.zeros([nbr_filters,patch_size,patch_size])

    for filt in range(0,nbr_filters):
        x = rand.randint(1,np.floor(patch_size/2)


