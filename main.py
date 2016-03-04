from utilities import init_poi
from feature_extraction import Feature
from amrafile import amrafile as af
import numpy as np


def main():
    
    target = '0015E'
    directory = '0030/'
    target_path = '/moria/data/DB/'+directory+target+'/wholebody_normalized_water_1_'+target+'.amra'
    prototype_path = '/home/hannes/DB/'+directory+target+'/prototypes'

    nbr_of_filters = 10
    patch_size = 7
    search_size = np.array([200,200,200])

    ''' Create feature object '''
    feature = Feature(nbr_of_filters, patch_size)

    reduced_data = init_poi(target_path,prototype_path, search_size, poi = 'T9')

    extracted_features = feature.feature_extraction(reduced_data, filter_type = 'Sobel')

if __name__ == "__main__":
    main()