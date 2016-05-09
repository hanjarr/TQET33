from utilities import Utils
from utilities import extract_weights
from feature_extraction import Feature
from forest_regression import RegressionForest
from forest_regression import run_forest
from sklearn.externals import joblib
import numpy as np
import json

class Module:

    with open("/home/hannes/code/git/multivariate/parameters.json") as json_file:
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

    #np.set_printoptions(threshold=np.nan)

    def pre_selection():

        ''' Empty arrays for concatenating features '''
        selection_features, ground_truth_z, ground_truth_y, ground_truth_x = np.array([]), np.array([]), np.array([]), np.array([])


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


            ''' Extract ground truth and features from filters'''
            ground_truth = utils.extract_ground_truth(reduced_mask)
            extracted_features = feature.haar_extraction(reduced_water, reduced_fat, [filter_bank,], [filter_parameters,])
            extracted_features = extracted_features[0]

            selection_features = np.vstack([extracted_features, selection_features]) if selection_features.size else extracted_features

            ground_truth_z = np.hstack([ground_truth_z, ground_truth[0]]) if ground_truth_z.size else ground_truth[0]
            ground_truth_y = np.hstack([ground_truth_y, ground_truth[1]]) if ground_truth_y.size else ground_truth[1]
            ground_truth_x = np.hstack([ground_truth_x, ground_truth[2]]) if ground_truth_x.size else ground_truth[2]

        selection_ground_truth = [ground_truth_z, ground_truth_y, ground_truth_x]

        ''' Select best filters according to feature importance measure'''
        output_filters, output_parameters = [], []

        for ind, ground_truth in enumerate(selection_ground_truth):

            selection = forest.feature_selection(selection_features, ground_truth, Module.selected_filters)

            ''' Extract best filters '''
            filters = [filter_bank[k] for k in selection]
            parameters = [filter_parameters[i] for i in selection]

            np.save('filter_bank_' + str(ind) + '.npy', filters)
            np.save('filter_parameters_' + str(ind) + '.npy', parameters)

            output_filters.append(filters)
            output_parameters.append(parameters)

        return output_filters, output_parameters

    def training(filter_banks, filter_parameters):

        ''' Empty arrays for concatenating features '''
        train_features_z, train_features_y, train_features_x, ground_truth_z, ground_truth_y, ground_truth_x = \
        np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

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

            ''' Extract features '''
            extracted_features = feature.haar_extraction(reduced_water, reduced_fat, filter_banks, filter_parameters)

            ''' Stack features, ground truth'''
            train_features_z = np.vstack([train_features_z, extracted_features[0]]) if train_features_z.size else extracted_features[0]
            train_features_y = np.vstack([train_features_y, extracted_features[1]]) if train_features_y.size else extracted_features[1]
            train_features_x = np.vstack([train_features_x, extracted_features[2]]) if train_features_x.size else extracted_features[2]

            ground_truth_z = np.hstack([ground_truth_z, ground_truth[0]]) if ground_truth_z.size else ground_truth[0]
            ground_truth_y = np.hstack([ground_truth_y, ground_truth[1]]) if ground_truth_y.size else ground_truth[1]
            ground_truth_x = np.hstack([ground_truth_x, ground_truth[2]]) if ground_truth_x.size else ground_truth[2]

        ''' Store ground truth and features in lists'''
        train_ground_truth = [ground_truth_z, ground_truth_y, ground_truth_x]
        train_features = [train_features_z, train_features_y, train_features_x]

        ''' Empty list for trained forest estimators, one for each orientation'''
        estimators = []

        ''' Create regression forest class object '''
        forest = RegressionForest(Module.nbr_of_trees, Module.max_features, Module.bootstrap)

        for ind, features in enumerate(train_features):

            ''' Generate trained forest '''
            estimator = forest.generate_forest(features, train_ground_truth[ind])

            ''' Save forest estimator to file'''
            joblib.dump(estimator, 'RegressionForest_' + str(ind) + '.pkl', compress=1)

            estimators.append(estimator)

        return estimators

    def testing(estimators, filter_banks, filter_parameters):

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

            ''' Extract testing grids'''
            position_grids = utils.extract_grids(reduced_mask)

            ''' Create feature object '''
            feature = Feature(Module.selected_filters, Module.patch_size)

            ''' Extract testing features '''
            test_features = feature.haar_extraction(reduced_water, reduced_fat, filter_banks, filter_parameters)

            ''' Run test data through forest '''
            regressions = []
            for estimator, features in zip(estimators, test_features):

                regression = run_forest(estimator, features)
                regression = [int(round(n, 0)) for n in regression]

                regressions.append(regression)

            reg_poi = utils.regression_voting(regressions, position_grids)
            print(reg_poi)

            reg_voxel_diff, ncc_voxel_diff, reg_diff = utils.error_measure(reg_poi, ncc_poi)
            print(reg_diff)

            ''' Plot the regression map '''
    
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