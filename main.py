from utilities import Utils
from utilities import load_prototypes
from feature_extraction import Feature
from forest_regression import RegressionForest
from sklearn.externals import joblib
from amrafile import amrafile as af
from tempfile import TemporaryFile
import json
import numpy as np


def main():

    with open("/home/hannes/code/git/parameters.json") as json_file:
        json_data = json.load(json_file)

    directory = json_data.get("directory")
    train_targets = json_data.get("train_targets")
    prototype_path = json_data.get("prototype_path")
    select_target = '0019A'
    test_target = '0019B'

    ''' choose POI '''
    poi = json_data.get('poi')

    ''' Haar feature parameters '''
    search_size = np.array(json_data.get('search_size'))
    nbr_of_filters = json_data.get('nbr_of_filters')
    selected_filters = json_data.get('selected_filters')
    patch_size = json_data.get('patch_size')

    ''' Forest parameters '''
    nbr_of_trees = json_data.get('nbr_of_trees')
    nbr_of_trees_select = json_data.get('nbr_of_trees_select')

    max_features = json_data.get('max_features')
    max_features_select = json_data.get('max_features_select')
    bootstrap = json_data.get('bootstrap')

    ''' Search extension for cross correlation''' 
    extension = json_data.get('extension')

    ''' Empty arrays for concatenating features '''
    train_features, train_ground_truth = np.array([]), np.array([])

    ''' Create feature object '''
    feature = Feature(nbr_of_filters, patch_size)

    ''' Generate filter bank '''
    filter_bank, filter_parameters = feature.generate_haar()

    #''' Load prototype data and POI positions'''
    #prototype_data, prototype_pois = load_prototypes(prototype_path, poi)


    ''' ------------------------------------------Feature selection to reduce dimensionality-----------------------------------------------------'''

    target_path = '/moria/data/DB/'+directory+select_target+'/wholebody_normalized_water_1_'+select_target+'.amra'

    ''' Create utils class object '''
    utils = Utils(target_path, search_size, extension, poi)

    #''' Extract reduced data '''
    #reduced_data, reduced_mask = utils.init_poi(prototype_data, prototype_pois)

    reduced_data, reduced_mask = utils.simple_search_reduction()

    ''' Extract ground truth '''
    ground_truth = utils.extract_ground_truth(reduced_mask)

    ''' Create regression forest class object '''
    forest = RegressionForest(nbr_of_trees_select, max_features_select, bootstrap)

    selection_features = feature.feature_extraction(reduced_data, filter_bank, filter_parameters)

    selection = forest.feature_selection(selection_features, ground_truth, selected_filters)

    filter_bank = filter_bank[selection,:,:]
    filter_parameters = [filter_parameters[i] for i in selection]

    ''' Create feature object '''
    feature = Feature(selected_filters, patch_size)

    del selection_features

    np.save('filter_bank.npy',filter_bank)
    np.save('filter_parameters', filter_parameters)


    for target in train_targets:
        target_path = '/moria/data/DB/'+directory+target+'/wholebody_normalized_water_1_'+target+'.amra'
        #prototype_path = '/home/hannes/DB/'+directory+target+'/prototypes'

        print(target)

        ''' Create utils class object '''
        utils = Utils(target_path, search_size, extension, poi)

        #''' Extract reduced data '''
        #reduced_data, reduced_mask = utils.init_poi(prototype_data, prototype_pois)

        ''' Init POI as just the ground truth + noise to reduce training time'''
        reduced_data, reduced_mask = utils.simple_search_reduction()

        ''' Extract ground truth '''
        ground_truth = utils.extract_ground_truth(reduced_mask)

        ''' Extract features '''
        extracted_features = feature.feature_extraction(reduced_data, filter_bank, filter_parameters)

        ''' Stack features and ground truth '''
        train_features = np.vstack([train_features, extracted_features]) if train_features.size else extracted_features
        train_ground_truth = np.hstack([train_ground_truth, ground_truth]) if train_ground_truth.size else ground_truth


    ''' Create regression forest class object '''
    forest = RegressionForest(nbr_of_trees, max_features, bootstrap)

    ''' Generate trained forest '''
    estimators = forest.generate_forest(train_features, train_ground_truth)

    ''' Save forest to file'''
    joblib.dump(estimators, 'RegressionForest.pkl', compress = 1)

if __name__ == "__main__":
    main()