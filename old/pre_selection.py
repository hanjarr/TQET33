from utilities import Utils
from feature_extraction import Feature
from forest_regression import RegressionForest
from sklearn.externals import joblib
from amrafile import amrafile as af
from tempfile import TemporaryFile
import json
import numpy as np


def pre_selection(json_data):

    directory = json_data['directory']
    protoytpe_path = json_data['prototype_path']
    selection_targets = json_data['selection_targets']

    search_size = np.array(json_data['search_size'])
    nbr_of_filters = json_data['nbr_of_filters']
    patch_size = json_data['patch_size']

    max_features_select = json_data.get('max_features_select')
    bootstrap = json_data.get('bootstrap')

    ''' choose POI '''
    poi = json_data.get('poi')

    ''' Search extension for cross correlation''' 
    extension = json_data.get('extension')


    ''' Empty arrays for concatenating features '''
    selection_features, selection_ground_truth = np.array([]), np.array([])

    ''' Create feature object '''
    feature = Feature(nbr_of_filters, patch_size)

    ''' Generate filter bank '''
    filter_bank, filter_parameters = feature.generate_haar()

    ''' Create regression forest class object '''
    forest = RegressionForest(nbr_of_trees_select, max_features_select, bootstrap)

    ''' Load prototype data and POI positions'''
    prototype_data, prototype_pois = load_prototypes(prototype_path, poi)


    ''' ------------------------------------------Feature selection to reduce dimensionality-----------------------------------------------------'''

    for target in selection_targets:
        target_path = '/moria/data/DB/'+directory+target+'/wholebody_normalized_water_1_'+target+'.amra'

        ''' Create utils class object '''
        utils = Utils(target_path, search_size, extension, poi)

        reduced_data, reduced_mask = utils.train_reduction()

        ''' Extract ground truth '''
        ground_truth = utils.extract_ground_truth(reduced_mask)
        extracted_features = feature.feature_extraction(reduced_data, filter_bank, filter_parameters)

        selection_features = np.vstack([extracted_features, selection_features]) if selection_features.size else extracted_features
        selection_ground_truth = np.hstack([selection_ground_truth, ground_truth]) if selection_ground_truth.size else ground_truth

    selection = forest.feature_selection(selection_features, selection_ground_truth, selected_filters)

    filter_bank = filter_bank[selection,:,:]
    filter_parameters = [filter_parameters[i] for i in selection]

    np.save('filter_bank.npy',filter_bank)
    np.save('filter_parameters', filter_parameters)

    return filter_bank, filter_paramters