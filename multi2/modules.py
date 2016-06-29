from utilities import Utils
from feature_extraction import Feature
from regression import Regression, run_estimator
from sklearn.externals import joblib
from sklearn import preprocessing
import numpy as np
import json

class Module:

    def __init__(self, parameters):
        with open(parameters) as json_file:
            json_data = json.load(json_file)

        self._directory = json_data['directory']
        self._selection_targets = json_data['selection_targets']
        self._train_targets = json_data["train_targets"]
        self._test_targets = json_data["test_targets"]

        self._search_size = np.array(json_data['search_size'])
        self._selected_filters = json_data.get('selected_filters')
        self._nbr_of_filters = json_data['nbr_of_filters']
        self._patch_size = json_data['patch_size']

        self._max_features_select = json_data['max_features_select']
        self._max_features = json_data['max_features']
        self._nbr_of_trees_select = json_data['nbr_of_trees_select']
        self._nbr_of_trees = json_data['nbr_of_trees']
        self._bootstrap = json_data['bootstrap']

        self._poi = json_data.get('poi')
        self._extension = json_data.get('extension')

        self._mean_dev = np.array(json_data.get('mean_dev'))
        self._mean_std = np.array(json_data.get('mean_std'))

    def pre_selection(self):

        ''' Empty arrays for concatenating features '''
        selection_features, selection_ground_truth = np.array([]), np.array([])

        ''' Create feature object '''
        feature = Feature(self._nbr_of_filters, self._patch_size)

        ''' Generate filter bank '''
        filter_bank, filter_parameters = feature.generate_haar_()

        ''' Create regression class object '''
        regressor = Regression(self._nbr_of_trees_select, self._max_features_select, self._bootstrap)


        ''' Extract data to do filter selection with '''
        for target in self._selection_targets:
            print(target)

            ''' Create utils class object '''
            utils = Utils(self._directory, target, self._search_size, self._extension, self._poi)

            ''' Extract reduced data from fat and water signal'''
            reduced_water, reduced_fat, reduced_mask = utils.train_reduction(self._mean_dev, self._mean_std)

            ''' Convolve with sobel filters'''
            sobel_water, sobel_fat = feature.sobel_extraction(reduced_water, reduced_fat)

            ''' Extract ground truth '''
            ground_truth = utils.extract_ground_truth(reduced_mask)

            water_features, fat_features = feature.haar_extraction(reduced_water, reduced_fat, filter_bank, filter_parameters)

            extracted_features = water_features

            selection_features = np.vstack([extracted_features, selection_features]) if selection_features.size else extracted_features
            selection_ground_truth = np.vstack([selection_ground_truth, ground_truth]) if selection_ground_truth.size else ground_truth

        ''' Select best filters '''
        selection = regressor.feature_selection(selection_features, selection_ground_truth, self._selected_filters)

        ''' Extract best filters '''
        filter_bank = [filter_bank[k] for k in selection]
        filter_parameters = [filter_parameters[i] for i in selection]

        np.save('filter_bank.npy',filter_bank)
        np.save('filter_parameters', filter_parameters)

        return filter_bank, filter_parameters

    def training(self, filter_bank, filter_parameters):

        ''' Empty arrays for concatenating features '''
        train_features, train_ground_truth = np.array([]), np.array([])

        ''' Create feature object '''
        feature = Feature(self._selected_filters, self._patch_size)

        for target in self._train_targets:
            print(target)

            ''' Create utils class object '''
            utils = Utils(self._directory, target, self._search_size, self._extension, self._poi)

            ''' Init POI as just the ground truth + noise to reduce training time'''
            reduced_water, reduced_fat, reduced_mask = utils.train_reduction(self._mean_dev, self._mean_std)

            ''' Convolve with sobel filters'''
            sobel_water, sobel_fat = feature.sobel_extraction(reduced_water, reduced_fat)

            ''' Extract ground truth '''
            ground_truth = utils.extract_ground_truth(reduced_mask)

            water_features, fat_features = feature.haar_extraction(reduced_water, reduced_fat, filter_bank, filter_parameters)

            extracted_features = water_features

            ''' Stack features, ground truth and weights'''
            train_features = np.vstack([train_features, extracted_features]) if train_features.size else extracted_features
            train_ground_truth = np.vstack([train_ground_truth, ground_truth]) if train_ground_truth.size else ground_truth


        ''' Create regression class object '''
        regressor = Regression(self._nbr_of_trees, self._max_features, self._bootstrap)

        ''' Generate trained regressor '''
        estimators = regressor.generate_estimator(train_features, train_ground_truth)

        ''' Save forest to file'''
        joblib.dump(estimators, 'RegressionForest.pkl', compress=1)

        return estimators

    def testing(self, estimators, filter_bank, filter_parameters):

        ''' Empty list for storing displacement from target POI'''
        reg_error, ncc_error, reg_voxel_error, ncc_voxel_error = [], [], np.array([]), np.array([]);

        ''' Create feature object '''
        feature = Feature(self._selected_filters, self._patch_size)

        for target in self._test_targets:
            print(target)

            prototype_path = '/media/hannes/localDrive/DB/' + self._directory + target + '/prototypes'

            ''' Create utils class object '''
            utils = Utils(self._directory, target, self._search_size, self._extension, self._poi)

            ''' Load prototype data and POI positions'''
            prototype_data, prototype_pois = utils.load_prototypes(prototype_path)

            ''' Init POI as just the ground truth + noise to reduce training time'''
            reduced_water, reduced_fat, reduced_mask, ncc_diff, ncc_poi = utils.init_poi(prototype_data, prototype_pois)

            ''' Convolve with sobel filters'''
            sobel_water, sobel_fat = feature.sobel_extraction(reduced_water, reduced_fat)

            ''' Extract testing ground truth '''
            ground_truth = utils.extract_ground_truth(reduced_mask)

            ''' Extract testing features '''
            water_features, fat_features = feature.haar_extraction(reduced_water, reduced_fat, filter_bank, filter_parameters)

            test_features =  water_features

            ''' Extract testing grids'''
            position_grids = utils.extract_grids(reduced_mask)

            ''' Run test data through regressor '''
            regressions = run_estimator(estimators, test_features)

            regression = regressions["Regression forest"]

            reg_poi, voting_map = utils.regression_voting(regression, position_grids)
            print(reg_poi)

            reg_voxel_diff, ncc_voxel_diff, reg_diff = utils.error_measure(reg_poi, ncc_poi)
            print(reg_diff)

            #utils.plot_regression(reg_poi, voting_map)

            ''' Save deviations from true POI'''
            reg_error.append(reg_diff)
            ncc_error.append(ncc_diff)

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