from utilities import Utils
from feature_extraction import Feature
from forest_regression import run_forest
from forest_regression import RegressionForest
from sklearn.externals import joblib
from amrafile import amrafile as af
import json
import numpy as np

with open("/home/hannes/code/git/parameters.json") as json_file:
    json_data = json.load(json_file)

directory = json_data.get("directory")
train_targets = json_data.get("train_targets")
prototype_path = json_data.get("prototype_path")
test_target = '0019A'


''' choose POI '''
poi = json_data.get('poi')

''' Haar feature parameters '''
search_size = np.array(json_data.get('search_size'))
nbr_of_filters = json_data.get('nbr_of_filters')
patch_size = json_data.get('patch_size')

''' Search extension for cross correlation''' 
extension = json_data.get('extension')

target_path = '/moria/data/DB/'+directory+test_target+'/wholebody_normalized_water_1_'+test_target+'.amra'

''' Load trained regression forest '''
estimators = joblib.load('/home/hannes/code/trained/test2/RegressionForest.pkl') 

filter_bank = np.load('/home/hannes/code/trained/test2/filter_bank.npy')
filter_parameters = np.load('/home/hannes/code/trained/test2/filter_parameters.npy')

''' Create utils class object '''
utils = Utils(target_path, search_size, extension, poi)

#''' Extract reduced test data '''
#reduced_data, reduced_mask = utils.init_poi(prototype_path)

''' Init POI as just the ground truth + noise to reduce training time'''
reduced_data, reduced_mask = utils.simple_search_reduction()

''' Extract testing ground truth '''
ground_truth = utils.extract_ground_truth(reduced_mask)

''' Create feature object '''
feature = Feature(nbr_of_filters, patch_size)

''' Extract testing features '''
test_features = feature.feature_extraction(reduced_data, filter_bank, filter_parameters)

''' Run test data through forest '''
regression = run_forest(estimators, test_features)

print(np.amin(regression))
print(regression.argmin())

poi_pos = utils.extract_poi_position(regression, reduced_mask)

print(poi_pos)