from utilities import Utils
from utilities import extract_weights
from feature_extraction import Feature
from forest_regression import RegressionForest
from forest_regression import run_forest
from sklearn.externals import joblib
import numpy as np
import json

class Module:
    with open("/home/hannes/code/git/parameters.json") as json_file:
        json_data = json.load(json_file)

    directory = json_data['directory']
    selection_targets = json_data['selection_targets']
    train_targets = json_data["train_targets"]
    test_targets = json_data["test_targets"]


    search_size = np.array(json_data['search_size'])
    selected_filters = json_data.get('selected_filters')
    nbr_of_filters = json_data['nbr_of_filters']
    patch_size = json_data['patch_size']

    max_features_select = json_data['max_features_select']
    max_features = json_data['max_features']
    nbr_of_trees_select = json_data['nbr_of_trees_select']
    nbr_of_trees = json_data['nbr_of_trees']
    bootstrap = json_data['bootstrap']

    ''' choose POI '''
    poi = json_data.get('poi')

    ''' Search extension for cross correlation''' 
    extension = json_data.get('extension')

    def pre_selection():

        ''' Empty arrays for concatenating features '''
        selection_features, selection_ground_truth, selection_weights = np.array([]), np.array([]), np.array([])


        ''' Create feature object '''
        feature = Feature(Module.nbr_of_filters, Module.patch_size)

        ''' Generate filter bank '''
        filter_bank, filter_parameters = feature.generate_haar_()

        ''' Create regression forest class object '''
        forest = RegressionForest(Module.nbr_of_trees_select, Module.max_features_select, Module.bootstrap)


        ''' Extract data to do filter selection with '''
        for target in Module.selection_targets:

            ''' Create utils class object '''
            utils = Utils(Module.directory, target, Module.search_size, Module.extension, Module.poi)

            ''' Extract reduced data from fat and water signal'''
            reduced_water, reduced_fat, reduced_mask = utils.train_reduction()

            ''' Extract ground truth '''
            #ground_truth = np.array(2*list(utils.extract_ground_truth(reduced_mask)))
            ground_truth = utils.extract_ground_truth(reduced_mask)
            water_features, fat_features = feature.haar_extraction(reduced_water, reduced_fat, filter_bank, filter_parameters)

            #''' Extract weights '''
            #weights = extract_weights(ground_truth)

            ''' Using both water and fat signal'''
            #extracted_features = np.vstack([water_features, fat_features])

            extracted_features = water_features

            selection_features = np.vstack([extracted_features, selection_features]) if selection_features.size else extracted_features
            selection_ground_truth = np.hstack([selection_ground_truth, ground_truth]) if selection_ground_truth.size else ground_truth
            #selection_weights = np.hstack([selection_weights, weights]) if selection_weights.size else weights


        ''' Select best filters '''
        selection = forest.feature_selection(selection_features, selection_ground_truth, Module.selected_filters)#, selection_weights)

        ''' Extract best filters '''
        filter_bank = [filter_bank[k] for k in selection]
        filter_parameters = [filter_parameters[i] for i in selection]

        np.save('filter_bank.npy',filter_bank)
        np.save('filter_parameters', filter_parameters)

        return filter_bank, filter_parameters

    def training(filter_bank, filter_parameters):

        ''' Empty arrays for concatenating features '''
        train_features, train_ground_truth, train_weights = np.array([]), np.array([]), np.array([])

        ''' Create feature object '''
        feature = Feature(Module.selected_filters, Module.patch_size)

        for target in Module.train_targets:
            print(target)

            ''' Create utils class object '''
            utils = Utils(Module.directory, target, Module.search_size, Module.extension, Module.poi)

            ''' Init POI as just the ground truth + noise to reduce training time'''
            reduced_water, reduced_fat, reduced_mask = utils.train_reduction()

            ''' Extract ground truth '''
            ground_truth = utils.extract_ground_truth(reduced_mask)

            #''' Extract weights '''
            #weights = extract_weights(ground_truth)

            ''' Extract features '''
            water_features, fat_features = feature.haar_extraction(reduced_water, reduced_fat, filter_bank, filter_parameters)

            ''' Concatenate horizontally for using both water and fat (feature dimension)'''
            #extracted_features = np.hstack([water_features, fat_features])

            extracted_features = water_features

            ''' Stack features, ground truth and weights'''
            train_features = np.vstack([train_features, extracted_features]) if train_features.size else extracted_features
            train_ground_truth = np.hstack([train_ground_truth, ground_truth]) if train_ground_truth.size else ground_truth
            #train_weights = np.hstack([train_weights, weights]) if train_weights.size else weights


        ''' Create regression forest class object '''
        forest = RegressionForest(Module.nbr_of_trees, Module.max_features, Module.bootstrap)

        ''' Generate trained forest '''
        estimators = forest.generate_forest(train_features, train_ground_truth)#, train_weights)

        ''' Save forest to file'''
        joblib.dump(estimators, 'RegressionForest.pkl', compress=1)

        return estimators

    def testing(estimators, filter_bank, filter_parameters):

        ''' Empty list for storing displacement from target POI'''
        reg_error, ncc_error, reg_voxel_error, ncc_voxel_error = [], [], np.array([]), np.array([]);

        for target in Module.test_targets:

            prototype_path = '/media/hannes/localDrive/DB/' + Module.directory + target + '/prototypes'

            print(target)

            ''' Create utils class object '''
            utils = Utils(Module.directory, target, Module.search_size, Module.extension, Module.poi)

            ''' Load prototype data and POI positions'''
            prototype_data, prototype_pois = utils.load_prototypes(prototype_path)

            ''' Init POI as just the ground truth + noise to reduce training time'''
            reduced_water, reduced_fat, reduced_mask, ncc_diff, ncc_poi = utils.init_poi(prototype_data, prototype_pois)

            ''' Extract testing ground truth '''
            ground_truth = utils.extract_ground_truth(reduced_mask)

            ''' Create feature object '''
            feature = Feature(Module.selected_filters, Module.patch_size)

            ''' Extract testing features '''
            water_features, fat_features = feature.haar_extraction(reduced_water, reduced_fat, filter_bank, filter_parameters)

            ''' Concatenate horizontally for using both water and fat (feature dimension)'''
            #test_features = np.hstack([water_features, fat_features])

            test_features = water_features

            ''' Run test data through forest '''
            regression = run_forest(estimators, test_features)

            reg_poi = utils.estimate_poi_position(regression, reduced_mask)
            print(reg_poi)

            reg_voxel_diff, ncc_voxel_diff, reg_diff = utils.error_measure(reg_poi, ncc_poi)
            print(reg_diff)

            ''' Plot the regression map '''
            #utils.plot_regression(regression)
    
            reg_error.append(reg_diff)
            ncc_error.append(ncc_diff)

            #utils.plot_reduced(reduced_water, ncc_poi, reg_poi)

            reg_voxel_error = np.vstack([reg_voxel_error, reg_voxel_diff]) if reg_voxel_error.size else reg_voxel_diff
            ncc_voxel_error = np.vstack([ncc_voxel_error, ncc_voxel_diff]) if ncc_voxel_error.size else ncc_voxel_diff


        error = list(zip(reg_error,ncc_error))
        voxel_error = list(zip(reg_voxel_error, ncc_voxel_error))

        np.save('error.npy', error)
        np.save('voxel_error.npy', voxel_error)

        print(np.mean(reg_error))
        print(np.std(reg_error))

        print(np.mean(ncc_error))
        print(np.std(ncc_error))

        return error, voxel_error