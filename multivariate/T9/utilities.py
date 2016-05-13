
from amrafile import amrafile as af
from amracommon.analysis.registration import normalized_cross_correlation
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import time

class Utils:

    def __init__(self, directory, target, search_size, extension, poi='T9'):

        ''' Get target from .amra format and full paths to prototypes'''
        water_path = '/moria/data/DB/'+ directory + target + '/wholebody_normalized_water_1_' + target +'.amra'
        fat_path = '/moria/data/DB/'+ directory + target + '/wholebody_normalized_fat_1_' + target +'.amra'

        self._water_target = af.parse(water_path)
        self._fat_target = af.parse(fat_path)

        ''' Get target as numpy arrays'''
        self._water_data = self._water_target.data
        self._fat_data = self._fat_target.data

        ''' Size of target data'''
        self._target_size = self._water_data.shape

        ''' Target poi ground truth'''
        self._target_poi = np.array(self._water_target.get_poi(poi))

        ''' Extract target voxel sizes and reduced voxel space'''
        self._target_voxel_size = self._water_target.voxel_size()
        self._search_size = search_size
        self._reduced_size = np.round(search_size/np.array(self._target_voxel_size)).astype(int)

        ''' Selected poi ''' 
        self._poi = poi

        ''' Extension for correlation '''
        self._extension = extension


    def init_poi(self, prototype_data, prototype_pois):
        
        ''' Extract best poi according to ncc and the reduced data space'''    
        reduced_water, reduced_fat, reduced_mask, poi_index = self.test_reduction(prototype_data, prototype_pois)

        ''' Best poi according to highest ncc measure'''
        ncc_poi = prototype_pois[poi_index]

        #''' Check correlation in nearby region to adjust poi initialization'''
        #adjusted_poi = self.cross_correlation(reduced_prototype, ncc_poi)

        ''' Number of prototypes used'''
        nbr_of_prototypes = len(prototype_data)

        ''' Repmat the target poi to compare distances'''
        rep_target_poi = np.tile(self._target_poi, (nbr_of_prototypes,1))
        rep_target_size = np.tile(self._target_voxel_size, (nbr_of_prototypes,1))

        ''' Diff between target poi ground truth and every deformed prototype poi in mm'''
        poi_diff = list(np.sqrt(np.sum((abs(np.array(prototype_pois) - rep_target_poi)*rep_target_size)**2, axis=1)))
        ncc_diff = poi_diff[poi_index]

        ''' Differences sorted in ascending order'''
        sorted_diff = sorted(poi_diff)
        sorted_index = sorted_diff.index(poi_diff[poi_index])

        ''' Best poi to choose from prototypes '''
        prototype_poi_index = poi_diff.index(min(poi_diff))
        sorted_index_best = sorted_diff.index(poi_diff[prototype_poi_index])

        print('\n'+'Ground truth:' )
        print(self._target_poi)
        print('\n'+'Poi according to highest ncc reduced:')
        print(ncc_poi)
        print(ncc_diff)

        ''' Plot the reduced target with transformed pois'''
        #plot_reduced(self, reduced_target, reduced_prototype, ncc_poi)

        return reduced_water, reduced_fat, reduced_mask, ncc_diff, ncc_poi


    def test_reduction(self, prototype_data, prototype_pois):

        ''' Init empty lists for storing reduced prototypes, reduced target and ncc'''
        reduced_prototypes, reduced_targets, reduced_masks, ncc = [], [], [], []
        reduced_mask = np.zeros((self._target_size), dtype = bool)

        ''' Calculate ncc between reduced target and reduced prototype space. 
            Reduced spaces will be of size 2*reduced_size+1 to get an odd kernel and a well defined center point''' 

        for ind, poi in enumerate(prototype_pois):

            z_lower = poi[0] - self._reduced_size[0]
            z_upper = poi[0] + self._reduced_size[0]+1
            y_lower = poi[1] - self._reduced_size[1]
            y_upper = poi[1] + self._reduced_size[1]+1
            x_lower = poi[2] - self._reduced_size[2]
            x_upper = poi[2] + self._reduced_size[2]+1

            prototype = prototype_data[ind]

            ''' Extract reduced space from prototype and target'''
            reduced_prototype = prototype[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]
            reduced_target = self._water_data[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]

            ''' Create binary mask of the reduced space'''
            reduced_mask_copy = reduced_mask.copy()
            reduced_mask_copy[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper] = True

            ''' Append reduced numpy arrays to lists'''
            reduced_masks.append(reduced_mask_copy)

            ''' Calculate ncc and store in lists'''
            ncc.append(normalized_cross_correlation(reduced_prototype, reduced_target))

        ''' Find index of poi which corresponds to highest ncc'''
        poi_index = ncc.index(max(ncc))    

        ''' Extract reduced data corresponding to highest ncc'''
        reduced_mask = reduced_masks[poi_index]

        reduced_water = np.reshape(self._water_data[reduced_mask], (2*self._reduced_size+1))
        reduced_fat = np.reshape(self._fat_data[reduced_mask], (2*self._reduced_size+1))

        return reduced_water, reduced_fat, reduced_mask, poi_index


    def train_reduction(self):

        ''' Init empty lists for storing reduced prototypes, reduced target and ncc'''
        reduced_mask = np.zeros((self._target_size), dtype = bool)

        ''' Calculate ncc between reduced target and reduced prototype space. 
            Reduced spaces will be of size 2*reduced_size+1 to get an odd kernel and a well defined center point''' 

        reduced_size = self._reduced_size
        poi = self._target_poi + np.round(abs(np.random.normal(2, 1, 3))).astype(int)

        z_lower = poi[0] - self._reduced_size[0]
        z_upper = poi[0] + self._reduced_size[0]+1
        y_lower = poi[1] - self._reduced_size[1]
        y_upper = poi[1] + self._reduced_size[1]+1
        x_lower = poi[2] - self._reduced_size[2]
        x_upper = poi[2] + self._reduced_size[2]+1

        ''' Extract reduced space from prototype and target'''
        reduced_water = self._water_data[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]
        reduced_fat = self._water_data[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]


        ''' Create binary mask of the reduced space'''
        reduced_mask_copy = reduced_mask.copy()
        reduced_mask[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper] = True

        print('\n'+'Ground truth:' )
        print(self._target_poi)

        return reduced_water, reduced_fat, reduced_mask


    def cross_correlation(self, reduced_prototype, ncc_poi):

        ''' Extend the reduced search space for cross correlation'''
        reduced_size = self._reduced_size
        ext = self._extension

        z_lower = ncc_poi[0]-ext-reduced_size[0]
        z_upper = ncc_poi[0]+ext+reduced_size[0]+1
        y_lower = ncc_poi[1]-ext-reduced_size[1]
        y_upper = ncc_poi[1]+ext+reduced_size[1]+1
        x_lower = ncc_poi[2]-ext-reduced_size[2]
        x_upper = ncc_poi[2]+ext+reduced_size[2]+1

        correlation_target = self._target_data[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]
        correlation_matrix = np.zeros((2*ext+1,2*ext+1,2*ext+1))
        center = np.array([ext, ext, ext])

        ''' Extract array shapes'''
        kernel_size = correlation_target.shape
        prototype_size = reduced_prototype.shape

        '''Hideous for-loop for doing cross-correlation'''
        for z in range(0, kernel_size[0]-prototype_size[0]+1):
            for y in range(0, kernel_size[1]-prototype_size[1]+1):
                for x in range(0,kernel_size[2]-prototype_size[2]+1):
                    correlation_kernel = correlation_target[z:z+prototype_size[0],y:y+prototype_size[1],x:x+prototype_size[2]]
                    correlation_matrix[z,y,x] = normalized_cross_correlation(reduced_prototype, correlation_kernel)

        ''' Extract index of max ncc from correlation matrix'''
        max_index = np.unravel_index(correlation_matrix.argmax(), correlation_matrix.shape)

        ''' Calculate updated POI'''
        adjusted_poi = max_index - center + ncc_poi

        return adjusted_poi

    def extract_ground_truth(self, reduced_mask):

        voxel_size = self._target_voxel_size
        target_poi = self._target_poi
        target_size = self._target_size

        z_upper = target_size[0]-target_poi[0]
        z_lower = target_size[0]-z_upper
        y_upper = target_size[1]-target_poi[1]
        y_lower = target_size[1]-y_upper
        x_upper = target_size[2]-target_poi[2]
        x_lower = target_size[2]-x_upper


        ''' Extract arrays containing voxel distances to the target poi position '''
        z_voxel_dist, y_voxel_dist, x_voxel_dist = np.mgrid[-z_lower:z_upper,-y_lower:y_upper,-x_lower:x_upper]

        ''' Express in milimeters according to voxel sizes '''
        voxel_dist = [z_voxel_dist, y_voxel_dist, x_voxel_dist]

        reduced_voxel_dist = [dist[reduced_mask] for dist in voxel_dist]

        return reduced_voxel_dist

    def extract_grids(self, reduced_mask):

        target_size = self._target_size

        ''' Extract arrays containing voxel positions in reduced search space'''
        z_grid, y_grid, x_grid = np.mgrid[0:target_size[0], 0:target_size[1], 0:target_size[2]]

        grids = [z_grid, y_grid, x_grid]

        reduced_grids = [grid[reduced_mask] for grid in grids]

        return reduced_grids

    def regression_voting(self, regression, position_grids):

        voting = np.array(position_grids) - np.array(regression)
        reg_positions = list(zip(voting[0], voting[1], voting[2]))

        reg_pos = np.array(max(set(reg_positions), key=reg_positions.count))

        return reg_pos


    def estimate_poi_position(self, regression, reduced_mask):

        ''' Total size of the reduced space '''
        reduced_size = 2*self._reduced_size+1

        ''' Meshgrid of the total target size in z,y,x directions '''
        z_ind, y_ind, x_ind = np.mgrid[0:self._target_size[0], 0:self._target_size[1], 0:self._target_size[2]]
        mesh_indices = [z_ind, y_ind, x_ind]

        ''' Reduced mesh index arrays ''' 
        reduced_mesh_indices = [mesh[reduced_mask] for mesh in mesh_indices]

        ''' Arg min corresponds to position of most probable POI position ''' 
        poi_index = regression.argmin()

        ''' Extract the index of the POI from the meshes '''
        poi_pos = np.array([mesh[poi_index] for mesh in reduced_mesh_indices])

        return poi_pos

    def error_measure(self, reg_poi, ncc_poi):
        reg_voxel_diff = abs(self._target_poi - reg_poi)
        ncc_voxel_diff = abs(self._target_poi - ncc_poi)

        reg_diff = np.sqrt(sum((reg_voxel_diff*self._target_voxel_size)**2))

        return reg_voxel_diff, ncc_voxel_diff, reg_diff

    def plot_regression(self, regression):

        reduced_size = (2*self._reduced_size+1)

        min_pos = np.unravel_index(regression.argmin(), reduced_size)
        regression_map = np.reshape(regression, reduced_size)

        plt.figure(frameon =False)
        currentAxis = plt.gca()
        plt.imshow((regression_map[:, min_pos[1], :]), plt.get_cmap('jet'), origin='lower')
        plt.autoscale(False)
        plt.colorbar()
        plt.plot(min_pos[2], min_pos[0], marker='o', color='g')
        #plt.plot(reg_poi[2], reg_poi[0], marker='o', color='b')

        plt.figure(frameon =False)
        currentAxis = plt.gca()
        plt.imshow((regression_map[:, : , min_pos[2]]), plt.get_cmap('jet'), origin='lower')
        plt.autoscale(False)
        plt.colorbar()
        plt.plot(min_pos[1], min_pos[0], marker='o', color='g')

        plt.figure(frameon =False)
        currentAxis = plt.gca()
        plt.imshow((regression_map[min_pos[0],:, :]), plt.get_cmap('jet'), origin='lower')
        plt.autoscale(False)
        plt.colorbar()
        plt.plot(min_pos[2], min_pos[1], marker='o', color='g')

        plt.show()


    def plot_reduced(self, reduced_target, ncc_poi, reg_poi):

        reduced_size = self._reduced_size

        # pos = (ncc_poi[2]+3, ncc_poi[0]+3)

        # plt.figure(frameon =False)
        # currentAxis = plt.gca()
        # plt.imshow((self._water_data[:, self._target_poi[1],:]), plt.get_cmap('gray'), origin='lower')
        # plt.autoscale(False)
        # plt.plot(pos[0], pos[1], marker='o', color='r')
        # currentAxis.add_patch(Rectangle((ncc_poi[2]-reduced_size[2],\
        # ncc_poi[0]-reduced_size[0]), reduced_size[2]*2+1, reduced_size[0]*2+1, fill=None, edgecolor="blue"))

        # currentAxis.add_patch(Rectangle((pos[0]-5,\
        # pos[1]-5), 3, 5, fill=None, edgecolor="green"))

        # currentAxis.add_patch(Rectangle((pos[0]+2,\
        # pos[1]+2), 5, 3, fill=None, edgecolor="red"))

        ''' Plot reduced area boxes around best poi and predicted poi'''

        plt.figure(frameon =False)
        currentAxis = plt.gca()
        plt.imshow((self._water_data[:, self._target_poi[1],:]), plt.get_cmap('gray'), origin='lower')
        plt.autoscale(False)
        plt.plot(self._target_poi[2], self._target_poi[0], marker='o', color='g')
        plt.plot(ncc_poi[2], ncc_poi[0], marker='o', color='r')
        plt.plot(reg_poi[2], reg_poi[0], marker='o', color='b')
        currentAxis.add_patch(Rectangle((ncc_poi[2]-reduced_size[2],\
        ncc_poi[0]-reduced_size[0]), reduced_size[2]*2+1, reduced_size[0]*2+1, fill=None, edgecolor="blue"))

        plt.figure(frameon =False)
        currentAxis = plt.gca()
        plt.imshow((self._water_data[:,:,self._target_poi[2]]), plt.get_cmap('gray'), origin='lower')
        plt.autoscale(False)
        plt.plot(self._target_poi[1], self._target_poi[0], marker='o', color='g')
        plt.plot(ncc_poi[1], ncc_poi[0], marker='o', color='r')
        plt.plot(reg_poi[1], reg_poi[0], marker='o', color='b')
        currentAxis.add_patch(Rectangle((ncc_poi[1]-reduced_size[1],\
        ncc_poi[0]-reduced_size[0]), reduced_size[1]*2+1, reduced_size[0]*2+1, fill=None, edgecolor="blue"))

        plt.figure(frameon =False)
        currentAxis = plt.gca()
        plt.imshow((self._water_data[self._target_poi[0],:,:]), plt.get_cmap('gray'), origin='lower')
        plt.autoscale(False)
        plt.plot(self._target_poi[2], self._target_poi[1], marker='o', color='g')
        plt.plot(ncc_poi[2], ncc_poi[1], marker='o', color='r')
        plt.plot(reg_poi[2], reg_poi[1], marker='o', color='b')
        currentAxis.add_patch(Rectangle((ncc_poi[2]-reduced_size[2],\
        ncc_poi[1]-reduced_size[1]), reduced_size[2]*2+1, reduced_size[1]*2+1, fill=None, edgecolor="blue"))

        plt.show()


    def load_prototypes(self, prototype_path):

        ''' List with paths to all prototypes'''
        prototype_paths = [join(prototype_path, f) for f in listdir(prototype_path) if isfile(join(prototype_path, f))]

        ''' Init empty lists for prototype signals and pois'''
        prototypes, prototype_pois = [], []


        ''' Get poi positions from prototypes'''
        for index, prototype in enumerate(prototype_paths):
            prototypes.append(af.parse(prototype))
            prototype_pois.append(prototypes[index].get_poi(self._poi))

        ''' Extract prototype data as numpy arrays'''
        prototype_data = [prototype.data for prototype in prototypes]


        return prototype_data, prototype_pois

def extract_weights(ground_truth):
    weights = 1-0.2*ground_truth/(np.amax(ground_truth))

    return weights

def plot_importances(estimators):

    ''' Extract feature importances '''
    importances = estimators["Regression forest"].feature_importances_

    ''' Get sorted indices of most important features'''
    std = np.std([tree.feature_importances_ for tree in estimators["Regression forest"].estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    indices = indices[0:30]

    '''Print the feature ranking'''
    print("Feature ranking:")

    for f in range(30):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    ''' Plot the feature importances of the forest '''
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(30), importances[indices],
        color="r", yerr=std[indices], align="center")
    plt.xticks(range(30), indices)
    plt.xlim([-1, 30])
    plt.show()

def plot_distribution(error, voxel_error):

    reg_error = [error[ind][0] for ind in range(len(error))]
    ncc_error = [error[ind][1] for ind in range(len(error))]

    reg_voxel_error = np.sum(np.array([voxel_error[ind][0] for ind in range(len(voxel_error))]),axis=1)
    ncc_voxel_error = np.sum(np.array([voxel_error[ind][1] for ind in range(len(voxel_error))]),axis=1)

    reg_hist = np.histogram(reg_error, range(15))
    ncc_hist = np.histogram(ncc_error, range(15))

    reg_voxel_hist = np.histogram(reg_voxel_error, range(10))
    ncc_voxel_hist = np.histogram(ncc_voxel_error, range(10))

    hist_data = [(reg_hist, ncc_hist),(reg_voxel_hist,ncc_voxel_hist)]

    for data in hist_data:

        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111,projection='3d')

        ax.bar(data[0][1][:-1], data[0][0], zs=0, zdir='y', color='b', alpha=0.6)
        ax.bar(data[1][1][:-1], data[1][0], zs=1, zdir='y', color='r', alpha=0.7)

        plt.autoscale(False)


        ax.set_xlabel('Deviation')
        ax.set_ylabel('Reg vs ncc')
        ax.set_zlabel('Frequency')
        plt.grid(True)

    plt.show()
