from utilities import Utils
from feature_extraction import Feature
from forest_regression import RegressionForest
from amrafile import amrafile as af
import numpy as np


def main():
   
    directory = '0030/' 
    train_targets = ['0015A','0015B']#,'0015C','0015D','0015E','0015F','00150','00151','00152','00154','00155']
    test_target = '0014A'

    ''' choose POI '''
    poi = 'LeftFemur'

    ''' Haar feature parameters '''
    nbr_of_filters = 100
    patch_size = 7
    search_size = np.array([50,40,40])

    ''' Forest parameters '''
    nbr_of_trees = 100
    nbr_of_features = 20
    bootstrap = True

    ''' Search extension for cross correlation''' 
    extension = 3

    ''' Empty arrays for concatenating features '''
    train_features, train_ground_truth = np.array([]), np.array([])

    for target in train_targets:
        target_path = '/moria/data/DB/'+directory+target+'/wholebody_normalized_water_1_'+target+'.amra'
        prototype_path = '/home/hannes/DB/'+directory+target+'/prototypes'

        ''' Create utils class object '''
        utils = Utils(target_path, search_size, extension, poi)

        ''' Extract reduced data '''
        reduced_data, reduced_mask = utils.init_poi(prototype_path)

        ''' Extract ground truth '''
        ground_truth = utils.extract_ground_truth(reduced_mask)

        ''' Create feature object '''
        feature = Feature(nbr_of_filters, patch_size)

        ''' Extract features '''
        extracted_features = feature.feature_extraction(reduced_data, filter_type = 'Haar')

        ''' Stack features and ground truth '''
        train_features = np.vstack([train_features, extracted_features]) if train_features.size else extracted_features
        train_ground_truth = np.hstack([train_ground_truth, ground_truth]) if train_ground_truth.size else ground_truth

    forest = RegressionForest(nbr_of_trees, nbr_of_features, bootstrap)

    estimators = forest.generate_forest(train_features, train_ground_truth)
    regression = forest.run_forest(estimators, extracted_features)

    print(type(regression))
    print(regression.shape)
    print(np.amin(regression))
    print(regression.argmin())

    poi_pos = utils.extract_poi_position(regression, reduced_mask)

    print(poi_pos)



if __name__ == "__main__":
    main()