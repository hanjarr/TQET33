from utilities import init_poi
from feature_extraction import Feature
from amrafile import amrafile as af
import numpy as np


def main():
   
    directory = '0030/' 
    target = '00155'
    target_path = '/moria/data/DB/'+directory+target+'/wholebody_normalized_water_1_'+target+'.amra'
    prototype_path = '/home/hannes/DB/'+directory+target+'/prototypes'

    nbr_of_filters = 10
    patch_size = 7
    search_size = np.array([50,40,40])

    ''' Search extension for cross correlation''' 
    extension = 3

    ''' Create feature object '''
    feature = Feature(nbr_of_filters, patch_size)

    reduced_data = init_poi(target_path,prototype_path, search_size, extension, poi = 'LeftFemur')

    extracted_features = feature.feature_extraction(reduced_data, filter_type = 'Haar')

if __name__ == "__main__":
    main()