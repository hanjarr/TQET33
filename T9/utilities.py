
from amrafile import amrafile as af
from amracommon.analysis.registration import normalized_cross_correlation
from scipy.ndimage.filters import gaussian_filter
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import time

class Utils:

    def __init__(self, directory, target, search_size, poi):

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


    def init_poi(self, prototype_data, prototype_pois):
        
        ''' Extract best poi according to ncc and the reduced data space'''    
        reduced_water, reduced_fat, reduced_mask, poi_index = self.initialization(prototype_data, prototype_pois)

        ''' Best poi according to highest ncc measure'''
        ncc_poi = prototype_pois[poi_index]

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

        return reduced_water, reduced_fat, reduced_mask, ncc_diff, ncc_poi


    def initialization(self, prototype_data, prototype_pois):

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

            lowpass_prototype = gaussian_filter(reduced_prototype,0.5,mode='constant')
            lowpass_target = gaussian_filter(reduced_target,0.5,mode='constant')

            ''' Create binary mask of the reduced space'''
            reduced_mask_copy = reduced_mask.copy()
            reduced_mask_copy[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper] = True

            ''' Append reduced numpy arrays to lists'''
            reduced_masks.append(reduced_mask_copy)

            ''' Calculate ncc and store in lists'''
            ncc.append(normalized_cross_correlation(lowpass_prototype, lowpass_target))

        ''' Find index of poi which corresponds to highest ncc'''
        poi_index = ncc.index(max(ncc))    

        ''' Extract reduced data corresponding to highest ncc'''
        reduced_mask = reduced_masks[poi_index]

        reduced_water = np.reshape(self._water_data[reduced_mask], (2*self._reduced_size+1))
        reduced_fat = np.reshape(self._fat_data[reduced_mask], (2*self._reduced_size+1))

        return reduced_water, reduced_fat, reduced_mask, poi_index

    def test_reduction(self, poi, vector):

        ''' Init empty lists for storing reduced prototypes, reduced target and ncc'''
        reduced_targets, reduced_masks = [], []
        reduced_mask = np.zeros((self._target_size), dtype = bool)

        ''' Calculate ncc between reduced target and reduced prototype space. 
            Reduced spaces will be of size 2*reduced_size+1 to get an odd kernel and a well defined center point'''

        estimate_poi = poi - np.round(np.array(vector)/np.array(self._target_voxel_size)).astype(int)

        estimate_diff = np.sqrt(sum((abs(self._target_poi-estimate_poi)*self._target_voxel_size)**2))

        z_lower = estimate_poi[0] - self._reduced_size[0]
        z_upper = estimate_poi[0] + self._reduced_size[0]+1
        y_lower = estimate_poi[1] - self._reduced_size[1]
        y_upper = estimate_poi[1] + self._reduced_size[1]+1
        x_lower = estimate_poi[2] - self._reduced_size[2]
        x_upper = estimate_poi[2] + self._reduced_size[2]+1

            
        ''' Create binary mask of the reduced space'''
        reduced_mask[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper] = True

        reduced_water = np.reshape(self._water_data[reduced_mask], (2*self._reduced_size+1))
        reduced_fat = np.reshape(self._fat_data[reduced_mask], (2*self._reduced_size+1))

        print('\n'+'Ground truth:' )
        print(self._target_poi)
        print('\n'+'Poi according to vector addition:')
        print(estimate_poi)
        print(estimate_diff)

        return reduced_water, reduced_fat, reduced_mask, estimate_poi


    def extract_grids(self, reduced_mask):

        target_size = self._target_size

        ''' Extract arrays containing voxel positions in reduced search space'''
        z_grid, y_grid, x_grid = np.mgrid[0:target_size[0], 0:target_size[1], 0:target_size[2]]

        grids = [z_grid, y_grid, x_grid]

        reduced_grids = np.transpose(np.array([grid[reduced_mask]*self._target_voxel_size[ind] for ind, grid in enumerate(grids)]))

        return reduced_grids

    def regression_voting(self, regression, position_grids):

        voting_map = np.zeros((self._target_size))

        ''' Numpy array containing all votes'''
        voting = np.round((position_grids - regression)/self._target_voxel_size).astype(int)

        ''' List with all regression positions'''
        reg_positions = list(zip(voting[:,0], voting[:,1], voting[:,2]))
        
        ''' Extract Poi with most votes'''
        max_pos = np.array(max(set(reg_positions), key=reg_positions.count))

        print(max_pos)

        ''' Voting map for plotting'''
        for pos in reg_positions:
            voting_map[pos] += 1

        ''' Normalize voting map'''
        voting_map /= np.amax(voting_map)

        ''' Get all positions which have more votes than 95 percent of the max number of votes'''
        top_votes = np.array(np.where(voting_map > 0.95))
        top_votes = np.array(list(zip(top_votes[0,:], top_votes[1,:], top_votes[2,:])))

        print(top_votes)

        reg_poi = tuple(np.round(np.mean(top_votes,0)).astype(int))

        return reg_poi, voting_map


    def error_measure(self, reg_poi, estimate_poi):
        reg_voxel_diff = abs(self._target_poi - reg_poi)
        estimate_voxel_diff = abs(self._target_poi - estimate_poi)

        reg_diff = np.sqrt(sum((reg_voxel_diff*self._target_voxel_size)**2))
        estimate_diff = np.sqrt(sum((reg_voxel_diff*self._target_voxel_size)**2))

        return reg_voxel_diff, estimate_voxel_diff, reg_diff, estimate_diff

    def plot_regression(self, reg_poi, voting_map):

        #reduced_size = (2*self._reduced_size+1)

        #[z_reg, y_reg, x_reg] = [np.reshape(regression[:,ind], reduced_size) for ind in range(0,3)]
        #regression_map = np.sqrt(z_reg**2 + y_reg**2 + x_reg**2)

        z_lower = reg_poi[0] - self._reduced_size[0]
        z_upper = reg_poi[0] + self._reduced_size[0] + 1
        y_lower = reg_poi[1] - self._reduced_size[1]
        y_upper = reg_poi[1] + self._reduced_size[1] + 1
        x_lower = reg_poi[2] - self._reduced_size[2]
        x_upper = reg_poi[2] + self._reduced_size[2] + 1


        plt.figure(frameon =False)
        currentAxis = plt.gca()
        plt.imshow((voting_map[reg_poi[0], y_lower:y_upper, x_lower:x_upper]), plt.get_cmap('jet'), interpolation='nearest', origin='lower')
        plt.autoscale(False)
        plt.colorbar()

        plt.figure(frameon =False)
        currentAxis = plt.gca()
        plt.imshow((voting_map[:, reg_poi[1], :]), plt.get_cmap('jet'), interpolation='nearest', origin='lower')
        plt.autoscale(False)
        plt.colorbar()

        plt.figure(frameon =False)
        currentAxis = plt.gca()
        plt.imshow((voting_map[:,:,reg_poi[2]]), plt.get_cmap('jet'), interpolation='nearest', origin='lower')
        plt.autoscale(False)
        plt.colorbar()

        test = voting_map[reg_poi[0], y_lower:y_upper, x_lower:x_upper]
        xx, yy = np.mgrid[0:test.shape[0], 0:test.shape[1]]

        # create the figure
        fig = plt.figure(frameon = False)
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, test ,rstride=1, cstride=1, cmap=plt.cm.jet,
        linewidth=0)

        # plt.figure(frameon =False)
        # currentAxis = plt.gca()
        # plt.imshow((regression_map[min_pos[0],:, :]), plt.get_cmap('jet'), origin='lower')
        # plt.autoscale(False)
        # plt.colorbar()
        # plt.plot(min_pos[2], min_pos[1], marker='o', color='g')

        plt.show()

    def plot_multi_regression(self, regressions):

        reduced_size = (2*self._reduced_size+1)

        [z,y,x] = [np.reshape(regression, reduced_size) for regression in regressions]

        [z_grid, y_grid, x_grid] = np.mgrid[-self._reduced_size[0] : self._reduced_size[0]+1,\
        -self._reduced_size[1] : self._reduced_size[1]+1,
        -self._reduced_size[2]:self._reduced_size[2]+1]

        fig = plt.figure(frameon=False)
        ax = fig.gca(projection='3d')

        ax.quiver(x_grid, y_grid, z_grid, x, y, z)

        ax.view_init(elev=18, azim=30)
        ax.dist=8 

        plt.show()


    def plot_reduced(self, reduced_target, ncc_poi, reg_poi):

        reduced_size = self._reduced_size

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