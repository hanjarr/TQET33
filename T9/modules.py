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
        self._test_targets = json_data["test_targets"]
        self._search_size = np.array(json_data['search_size'])
        self._poi_list = json_data.get('poi_list')
        self._selected_filters = json_data.get('selected_filters')
        self._patch_size = json_data.get('patch_size')
        self._vectors = json_data.get('vectors')
        self.poi_pos = []


    def testing(self):

        ''' Empty list for storing displacement from target POI'''
        reg_error, estimate_error, reg_voxel_error, reg_pois = [], [], np.array([]), np.array([]);

        for target in self._test_targets:
            print(target)

            reg_diff_list = []

            for ind, poi in enumerate(self._poi_list):

                print('Current POI:')
                print(poi)

                ''' Create utils class object '''
                utils = Utils(self._directory, target, self._search_size, poi)

                ''' Create feature object '''
                feature = Feature(self._selected_filters, self._patch_size)

                filter_bank = np.load('/media/hannes/localDrive/trained/multi2/T9/'+ poi +'/filter_bank.npy')
                filter_parameters = np.load('/media/hannes/localDrive/trained/multi2/T9/'+ poi +'/filter_parameters.npy')
                estimator = joblib.load('/media/hannes/localDrive/trained/multi2/T9/'+ poi +'/RegressionForest.pkl')


                if poi == 'RightFemur':

                    prototype_path = '/media/hannes/localDrive/DB/' + self._directory + target + '/prototypes'

                    ''' Load prototype data and POI positions'''
                    prototype_data, prototype_pois = utils.load_prototypes(prototype_path)

                    ''' Init POI as just the ground truth + noise to reduce training time'''
                    reduced_water, reduced_fat, reduced_mask, ncc_diff, estimate_poi = utils.init_poi(prototype_data, prototype_pois)

                    self.poi_pos.append(estimate_poi)

                else:

                    reduced_water, reduced_fat, reduced_mask, estimate_poi = utils.test_reduction(self.poi_pos[-1], self._vectors[ind-1])

                ''' Extract testing features '''
                water_features, fat_features = feature.haar_extraction(reduced_water, reduced_fat, filter_bank, filter_parameters)

                test_features = water_features

                ''' Extract testing grids'''
                position_grids = utils.extract_grids(reduced_mask)

                ''' Run test data through regressor '''
                regressions = run_estimator(estimator, test_features)

                regression = regressions["Regression forest"]

                reg_poi, voting_map = utils.regression_voting(regression, position_grids)
                print(reg_poi)

                reg_diff, estimate_diff = utils.error_measure(reg_poi, estimate_poi)
                print(reg_diff)

                self.poi_pos.append(reg_poi)
                reg_diff_list.append(reg_diff)

                #utils.plot_regression(estimate_poi, reg_poi, voting_map)
                #utils.plot_reduced(reduced_water, estimate_poi, reg_poi)

            ''' Save deviations from true POI'''
            reg_error.append(reg_diff)
            estimate_error.append(estimate_diff)

            reg_pois = np.vstack([reg_pois, np.array(reg_diff_list)]) if reg_pois.size else np.array(reg_diff_list)

        error = list(zip(reg_error, estimate_error))

        np.save('error.npy', error)
        np.save('reg.npy', reg_pois)

        print(np.mean(reg_error))
        print(np.std(reg_error))

        print(np.mean(estimate_error))
        print(np.std(estimate_error))

        return error, reg_pois