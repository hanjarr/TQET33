from utilities import Utils
from utilities import extract_weights
from feature_extraction import Feature
from cnn_regression import NeuralNetwork, run_cnn
from sklearn.externals import joblib
import numpy as np
import json

class Module:

    def __init__(self, parameters):
        with open(parameters) as json_file:
            json_data = json.load(json_file)

        self._directory = json_data['directory']
        self._train_targets = json_data["train_targets"]
        self._test_targets = json_data["test_targets"]
        self._search_size = np.array(json_data['search_size'])

        self._poi = json_data.get('poi')
        self._extension = json_data.get('extension')

        self._kernel_size = json_data.get('kernel_size')
        self._iterations = json_data.get('iterations') 
        self._learning_rate = json_data.get('learning_rate') 

    def training(self):

        ''' Empty arrays for concatenating features '''
        features_z, features_y, features_x, ground_truth_z, ground_truth_y, ground_truth_x = \
        np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])


        for target in self._train_targets:
            print(target)

            ''' Create utils class object '''
            utils = Utils(self._directory, target, self._search_size, self._extension, self._poi)

            ''' Init POI as just the ground truth + noise to reduce training time'''
            reduced_water, reduced_fat, reduced_mask = utils.train_reduction()

            ''' Extract ground truth'''
            ground_truth = utils.extract_ground_truth(reduced_mask)

            ''' Stack features, ground truth'''
            features_z = np.vstack([features_z, reduced_water]) if features_z.size else reduced_water
            features_y = np.hstack([features_y, reduced_water]) if features_y.size else reduced_water
            features_x = np.dstack([features_x, reduced_water]) if features_x.size else reduced_water

            ground_truth_z = np.hstack([ground_truth_z, ground_truth[0][:,0,0]]) if ground_truth_z.size else ground_truth[0][:,0,0]
            ground_truth_y = np.hstack([ground_truth_y, ground_truth[1][0,:,0]]) if ground_truth_y.size else ground_truth[1][0,:,0]
            ground_truth_x = np.hstack([ground_truth_x, ground_truth[2][0,0,:]]) if ground_truth_x.size else ground_truth[2][0,0,:]

        # print(np.shape(features_z))
        # print(np.shape(features_y))
        # print(np.shape(features_x))

        # print(np.shape(np.transpose(features_y,(1,2,0))))
        # print(np.shape(np.transpose(features_x,(2,0,1))))


        ''' Store ground truth and features in lists'''
        train_ground_truth = [ground_truth_z, ground_truth_y, ground_truth_x]
        train_features = [features_z, np.transpose(features_y,(1,0,2)), np.transpose(features_x,(2,0,1))]

        ''' Empty list for trained cnn estimators, one for each orientation'''
        estimators = []

        ''' Create regression cnn class object '''
        net = NeuralNetwork(self._kernel_size, self._iterations, self._learning_rate)

        for ind, train_feature in enumerate(train_features):

            ''' Generate trained cnn '''
            estimator = net.generate_cnn(train_feature, train_ground_truth[ind])

            ''' Save cnn estimator to file'''
            joblib.dump(estimator, 'Network_' + str(ind) + '.pkl', compress=1)

            estimators.append(estimator)

        return estimators

    def testing(self, estimators):

        ''' Empty list for storing displacement from target POI'''
        reg_error, ncc_error, reg_voxel_error, ncc_voxel_error = [], [], np.array([]), np.array([]);


        for target in self._test_targets:

            prototype_path = '/media/hannes/localDrive/DB/' + self._directory + target + '/prototypes'
            print(target)

            ''' Create utils class object '''
            utils = Utils(self._directory, target, self._search_size, self._extension, self._poi)

            ''' Load prototype data and POI positions'''
            prototype_data, prototype_pois = utils.load_prototypes(prototype_path)

            ''' Init POI as just the ground truth + noise to reduce training time'''
            reduced_water, reduced_fat, reduced_mask, ncc_diff, ncc_poi = utils.init_poi(prototype_data, prototype_pois)

            test_features = [reduced_water, np.transpose(reduced_water,(1,0,2)), np.transpose(reduced_water,(2,0,1))]

            #print(np.shape(features[1]))
            #print(np.shape(features[2]))

            ''' Extract testing grids'''
            position_vectors = utils.extract_grids(reduced_mask)


            ''' Run test data through cnn '''
            regressions = []
            for estimator, test_feature in zip(estimators, test_features):

                regression = run_cnn(estimator, test_feature)
                regression = np.ndarray.flatten(np.round(regression).astype(int))

                regressions.append(regression)

            reg_poi = utils.regression_voting(regressions, position_vectors)
            print(reg_poi)

            reg_voxel_diff, ncc_voxel_diff, reg_diff = utils.error_measure(reg_poi, ncc_poi)
            print(reg_diff)

            ''' Save deviations from true POI'''
            reg_error.append(reg_diff)
            ncc_error.append(ncc_diff)

            ''' Plot the regression map '''
            #utils.plot_multi_regression(regressions)

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