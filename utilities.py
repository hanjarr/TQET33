
from amrafile import amrafile as af
from amracommon.analysis.registration import normalized_cross_correlation
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

class Utils:

    def __init__(self, target_path, search_size, extension, poi='T9'):

        ''' Get target from .amra format and full paths to prototypes'''
        self._target = af.parse(target_path)

        ''' Get target and prototypes as numpy arrays'''
        self._target_data = self._target.data

        ''' Size of target data'''
        self._target_size = self._target_data.shape

        ''' Target poi ground truth'''
        self._target_poi = np.array(self._target.get_poi(poi))

        ''' Extract target voxel sizes and reduced voxel space'''
        self._target_voxel_size = self._target.voxel_size()
        self._search_size = search_size
        self._reduced_size = np.round(search_size/np.array(self._target_voxel_size)).astype(int)

        ''' Chosen_poi ''' 
        self._poi = poi

        ''' Extension for correlation '''
        self._extension = extension



    def init_poi(self, prototype_path):

        ''' List with paths to all prototypes'''
        prototype_paths = [join(prototype_path, f) for f in listdir(prototype_path) if isfile(join(prototype_path, f))]


        ''' Init empty lists for prototype signals and pois'''
        prototypes, prototype_pois = [], []

        start = time.time()
        print("Load prototypes")

        ''' Get poi positions from prototypes'''
        for index, prototype in enumerate(prototype_paths):
            prototypes.append(af.parse(prototype))
            prototype_pois.append(prototypes[index].get_poi(self._poi))

        ''' Extract prototype data as numpy arrays'''
        prototype_data = [prototype.data for prototype in prototypes]

        end = time.time()
        print(end - start)


        start = time.time()
        print("Extract best initial poi")
        
        ''' Extract best poi according to ncc and the reduced data space'''    
        reduced_target, reduced_prototype, reduced_mask, poi_index = self.search_reduction(prototype_data, prototype_pois)

        end = time.time()
        print(end - start)

        start = time.time()
        print("Cross correlation")

        ''' Best poi according to highest ncc measure'''
        ncc_poi = prototype_pois[poi_index]

        ''' Check correlation in nearby region to adjust poi initialization'''
        adjusted_poi = self.cross_correlation(reduced_prototype, ncc_poi)

        ''' Number of prototypes used'''
        nbr_of_prototypes = len(prototypes)

        ''' Repmat the target poi to compare distances'''
        rep_target_poi = np.tile(self._target_poi, (nbr_of_prototypes,1))
        rep_target_size = np.tile(self._target_voxel_size, (nbr_of_prototypes,1))

        ''' Diff between target poi ground truth and every deformed prototype poi in mm'''
        poi_diff = list(np.sum(abs(np.array(prototype_pois) - rep_target_poi)*rep_target_size, axis=1))

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
        print(poi_diff[poi_index])
        print(sorted_index)
        print('\n'+'Best prototype poi:' )
        print(prototype_pois[prototype_poi_index])
        print(poi_diff[prototype_poi_index])

        ''' Plot the reduced spaces and transformed pois'''
        #plot_reduced(self, reduced_target, reduced_prototype, ncc_poi)

        mean_poi = np.array(prototype_pois).mean(axis=0)

        end = time.time()
        print(end - start)

        return reduced_target, reduced_mask

    def plot_reduced(self, reduced_target, reduced_prototype, ncc_poi):

        reduced_size = self._reduced_size

        ''' Show reduced data in target'''    
        plt.figure()
        plt.subplot(211)
        plt.imshow((reduced_target[:,reduced_size[1],:]), plt.get_cmap('gray'),origin='lower')

        ''' Show reduced data in prototype'''
        plt.subplot(212)
        plt.imshow((reduced_prototype[:,reduced_size[1],:]), plt.get_cmap('gray'),origin='lower')

        ''' Plot reduced area boxes around best poi when calculating full ncc and reduced. 
            Green marker is true poi, blue reduced, red full'''
        plt.figure()
        currentAxis = plt.gca()
        plt.imshow((self._target_data[:, ncc_poi[1],:]), plt.get_cmap('gray'), origin='lower')
        plt.autoscale(False)
        plt.plot(self._target_poi[2], self._target_poi[0], marker='o', color='g')
        plt.plot(ncc_poi[2], ncc_poi[0], marker='o', color='b')
        #plt.plot(max_ncc_poi_full[2], max_ncc_poi_full[0], marker='o', color='r')
        currentAxis.add_patch(Rectangle((ncc_poi[2]-reduced_size[2],\
        ncc_poi[0]-reduced_size[0]), reduced_size[2]*2, reduced_size[0]*2, fill=None, edgecolor="blue"))
        #currentAxis.add_patch(Rectangle((max_ncc_poi_full[2]-reduced_size[2], max_ncc_poi_full[0]-reduced_size[0]), reduced_size[2]*2, reduced_size[0]*2, fill=None, edgecolor="red"))

        plt.figure()
        currentAxis = plt.gca()
        plt.imshow((self._target_data[:,:,ncc_poi[2]]), plt.get_cmap('gray'), origin='lower')
        plt.autoscale(False)
        plt.plot(self._target_poi[1], self._target_poi[0], marker='o', color='g')
        plt.plot(ncc_poi[1], ncc_poi[0], marker='o', color='b')
        #plt.plot(max_ncc_poi_full[1], max_ncc_poi_full[0], marker='o', color='r')
        currentAxis.add_patch(Rectangle((ncc_poi[1]-reduced_size[1],\
        ncc_poi[0]-reduced_size[0]), reduced_size[1]*2, reduced_size[0]*2, fill=None, edgecolor="blue"))
        #currentAxis.add_patch(Rectangle((max_ncc_poi_full[1]-reduced_size[1], max_ncc_poi_full[0]-reduced_size[0]), reduced_size[1]*2, reduced_size[0]*2, fill=None, edgecolor="red"))

        plt.figure()
        currentAxis = plt.gca()
        plt.imshow((self._target_data[ncc_poi[0],:,:]), plt.get_cmap('gray'), origin='lower')
        plt.autoscale(False)
        plt.plot(self._target_poi[2], self._target_poi[1], marker='o', color='g')
        plt.plot(ncc_poi[2], ncc_poi[1], marker='o', color='b')
        #plt.plot(max_ncc_poi_full[2], max_ncc_poi_full[1], marker='o', color='r')
        currentAxis.add_patch(Rectangle((ncc_poi[2]-reduced_size[2],\
        ncc_poi[1]-reduced_size[1]), reduced_size[2]*2, reduced_size[1]*2, fill=None, edgecolor="blue"))
        #currentAxis.add_patch(Rectangle((max_ncc_poi_full[2]-reduced_size[2], max_ncc_poi_full[1]-reduced_size[1]), reduced_size[2]*2, reduced_size[2]*2, fill=None, edgecolor="red"))


        # plt.subplot(212)
        # #currentAxis = plt.gca()
        # plt.imshow((prototype_data[poi_index][:,max_ncc_poi[1],:]), plt.get_cmap('gray'), origin='lower')
        # #plt.plot(max_ncc_poi[2],max_ncc_poi[0], marker='o', color='b')
        # #currentAxis.add_patch(Rectangle((max_ncc_poi[2]-reduced_size[2], max_ncc_poi[0]-reduced_size[0]), reduced_size[2]*2, reduced_size[0]*2, fill=None, edgecolor="blue"))

        plt.show()

    def search_reduction(self, prototype_data, prototype_pois):

        ''' Init empty lists for storing reduced prototypes, reduced target and ncc'''
        reduced_prototypes, reduced_targets, reduced_masks, ncc = [], [], [], []
        reduced_mask = np.zeros((self._target_size), dtype = bool)

        ''' Calculate ncc between reduced target and reduced prototype space. 
            Reduced spaces will be of size 2*reduced_size+1 to get an odd kernel and a well defined center point''' 

        reduced_size = self._reduced_size

        for ind, poi in enumerate(prototype_pois):

            z_lower = poi[0]-reduced_size[0]
            z_upper = poi[0]+reduced_size[0]+1
            y_lower = poi[1]-reduced_size[1]
            y_upper = poi[1]+reduced_size[1]+1
            x_lower = poi[2]-reduced_size[2]
            x_upper = poi[2]+reduced_size[2]+1

            prototype = prototype_data[ind]

            ''' Extract reduced space from prototype and target'''
            reduced_prototype = prototype[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]
            reduced_target = self._target_data[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]

            ''' Create binary mask of the reduced space'''
            reduced_mask_copy = reduced_mask.copy()
            reduced_mask_copy[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper] = True

            ''' Append reduced numpy arrays to lists'''
            reduced_prototypes.append(reduced_prototype)
            reduced_targets.append(reduced_target)
            reduced_masks.append(reduced_mask_copy)

            ''' Calculate ncc and store in lists'''
            ncc.append(normalized_cross_correlation(reduced_prototype, reduced_target))

        ''' Find index of poi which corresponds to highest ncc'''
        poi_index = ncc.index(max(ncc))    

        '''------------Poi positioned in middle of reduced_target-------------------'''
        #print(reduced_targets[0][reduced_size[0],reduced_size[1],reduced_size[2]])

        ''' Extract reduced data corresponding to highest ncc'''
        reduced_target = reduced_targets[poi_index]
        reduced_prototype = reduced_prototypes[poi_index]
        reduced_mask = reduced_masks[poi_index]


        return reduced_target, reduced_prototype, reduced_mask, poi_index


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
        z_voxel_dist, y_voxel_dist, x_voxel_dist = abs(np.mgrid[-z_lower:z_upper,-y_lower:y_upper,-x_lower:x_upper])

        ''' Express in milimeters according to voxel sizes '''
        voxel_dist = [z_voxel_dist, y_voxel_dist, x_voxel_dist]
        reduced_voxel_dist = [dist[reduced_mask] for dist in voxel_dist]
        dist = [reduced_voxel_dist[i]*voxel_size[i] for i in range(0,3)]

        ''' Vector containing ground truth distances from every voxel to target POI '''
        ground_truth = np.sqrt(dist[0]**2 + dist[1]**2 + dist[2]**2)

        return ground_truth

    def extract_poi_position(self, regression, reduced_mask):

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

