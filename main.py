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
    search_size = np.array([50,50,50])

    ''' Create feature object '''
    feature = Feature(nbr_of_filters, patch_size)

    reduced_data = init_poi(target_path,prototype_path, search_size, poi = 'RightFemur')

    #extracted_features = feature.feature_extraction(reduced_data, filter_type = 'Sobel')

if __name__ == "__main__":
    main()