
from amrafile import amrafile as af
from amracommon.analysis.registration import normalized_cross_correlation
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy import unravel_index
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def init_poi(target_path, prototype_path, search_size, extension, poi = 'T9'):

    ''' Get target from .amra format and full paths to prototypes'''
    target = af.parse(target_path)
    prototype_paths = [join(prototype_path, f) for f in listdir(prototype_path) if isfile(join(prototype_path, f))]

    ''' Extract target voxel sizes and reduced voxel space'''
    voxel_sizes = target.voxel_size()
    reduced_size = np.round(search_size/np.array(voxel_sizes)).astype(int)

    ''' Init empty lists for prototype signals and pois'''
    prototypes, prototype_pois = [], []

    ''' Target poi ground truth'''
    target_poi = np.array(target.get_poi(poi))

    ''' Get poi positions from prototypes'''
    for index, prototype in enumerate(prototype_paths):
        prototypes.append(af.parse(prototype))
        prototype_pois.append(prototypes[index].get_poi(poi))

    ''' Get target and prototypes as numpy arrays'''
    target_data = target.data
    prototype_data = [prototype.data for prototype in prototypes]
    
    ''' Extract best poi according to ncc and the reduced data space'''    
    reduced_target, reduced_prototype, poi_index = search_reduction(target_data, prototype_data, target_poi, prototype_pois, reduced_size)

    ''' Best poi according to highest ncc measure'''
    max_ncc_poi = prototype_pois[poi_index]

    adjusted_poi = cross_correlation(target_data, reduced_prototype, max_ncc_poi, reduced_size, extension)
    ''' Number of prototypes used'''
    nbr_of_prototypes = len(prototypes)

    ''' Repmat the target poi to compare distances'''
    rep_target_poi = np.tile(target_poi, (nbr_of_prototypes,1))
    rep_voxel_sizes = np.tile(target.voxel_size(), (nbr_of_prototypes,1))

    ''' Diff between target poi ground truth and every deformed prototype poi in mm'''
    poi_diff = list(np.sum(abs(np.array(prototype_pois) - rep_target_poi)*rep_voxel_sizes, axis=1))

    ''' Differences sorted in ascending order'''
    sorted_diff = sorted(poi_diff)
    sorted_index = sorted_diff.index(poi_diff[poi_index])

    ''' Best poi to choose from prototypes '''
    prototype_poi_index = poi_diff.index(min(poi_diff))
    sorted_index_best = sorted_diff.index(poi_diff[prototype_poi_index])

    print('\n'+'Ground truth:' )
    print(target_poi)
    print('\n'+'Poi according to highest ncc reduced:')
    print(max_ncc_poi)
    print(poi_diff[poi_index])
    print(sorted_index)
    print('\n'+'Best prototype poi:' )
    print(prototype_pois[prototype_poi_index])
    print(poi_diff[prototype_poi_index])

    ''' Plot the reduced spaces and transformed pois'''
    plot_reduced(target_data, reduced_target, reduced_prototype, max_ncc_poi, target_poi, reduced_size)

    mean_poi = np.array(prototype_pois).mean(axis=0)


    return reduced_target

def plot_reduced(target_data, reduced_target, reduced_prototype, max_ncc_poi, target_poi, reduced_size):

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
    plt.imshow((target_data[:, max_ncc_poi[1],:]), plt.get_cmap('gray'), origin='lower')
    plt.autoscale(False)
    plt.plot(target_poi[2], target_poi[0], marker='o', color='g')
    plt.plot(max_ncc_poi[2], max_ncc_poi[0], marker='o', color='b')
    #plt.plot(max_ncc_poi_full[2], max_ncc_poi_full[0], marker='o', color='r')
    currentAxis.add_patch(Rectangle((max_ncc_poi[2]-reduced_size[2],\
    max_ncc_poi[0]-reduced_size[0]), reduced_size[2]*2, reduced_size[0]*2, fill=None, edgecolor="blue"))
    #currentAxis.add_patch(Rectangle((max_ncc_poi_full[2]-reduced_size[2], max_ncc_poi_full[0]-reduced_size[0]), reduced_size[2]*2, reduced_size[0]*2, fill=None, edgecolor="red"))

    plt.figure()
    currentAxis = plt.gca()
    plt.imshow((target_data[:,:,max_ncc_poi[2]]), plt.get_cmap('gray'), origin='lower')
    plt.autoscale(False)
    plt.plot(target_poi[1], target_poi[0], marker='o', color='g')
    plt.plot(max_ncc_poi[1], max_ncc_poi[0], marker='o', color='b')
    #plt.plot(max_ncc_poi_full[1], max_ncc_poi_full[0], marker='o', color='r')
    currentAxis.add_patch(Rectangle((max_ncc_poi[1]-reduced_size[1],\
    max_ncc_poi[0]-reduced_size[0]), reduced_size[1]*2, reduced_size[0]*2, fill=None, edgecolor="blue"))
    #currentAxis.add_patch(Rectangle((max_ncc_poi_full[1]-reduced_size[1], max_ncc_poi_full[0]-reduced_size[0]), reduced_size[1]*2, reduced_size[0]*2, fill=None, edgecolor="red"))

    plt.figure()
    currentAxis = plt.gca()
    plt.imshow((target_data[max_ncc_poi[0],:,:]), plt.get_cmap('gray'), origin='lower')
    plt.autoscale(False)
    plt.plot(target_poi[2], target_poi[1], marker='o', color='g')
    plt.plot(max_ncc_poi[2], max_ncc_poi[1], marker='o', color='b')
    #plt.plot(max_ncc_poi_full[2], max_ncc_poi_full[1], marker='o', color='r')
    currentAxis.add_patch(Rectangle((max_ncc_poi[2]-reduced_size[2],\
    max_ncc_poi[1]-reduced_size[1]), reduced_size[2]*2, reduced_size[1]*2, fill=None, edgecolor="blue"))
    #currentAxis.add_patch(Rectangle((max_ncc_poi_full[2]-reduced_size[2], max_ncc_poi_full[1]-reduced_size[1]), reduced_size[2]*2, reduced_size[2]*2, fill=None, edgecolor="red"))


    # plt.subplot(212)
    # #currentAxis = plt.gca()
    # plt.imshow((prototype_data[poi_index][:,max_ncc_poi[1],:]), plt.get_cmap('gray'), origin='lower')
    # #plt.plot(max_ncc_poi[2],max_ncc_poi[0], marker='o', color='b')
    # #currentAxis.add_patch(Rectangle((max_ncc_poi[2]-reduced_size[2], max_ncc_poi[0]-reduced_size[0]), reduced_size[2]*2, reduced_size[0]*2, fill=None, edgecolor="blue"))

    plt.show()

def search_reduction(target_data, prototype_data, target_poi, prototype_pois, reduced_size):

    ''' Init empty lists for storing reduced prototypes, reduced target and ncc'''
    reduced_prototypes, reduced_targets, ncc = [], [], []

    ''' Calculate ncc between reduced target and reduced prototype space. 
        Reduced spaces will be of size 2*reduced_size+1 to get an odd kernel and a well defined mid point''' 

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
        reduced_target = target_data[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]

        ''' Append reduced numpy arrays to lists'''
        reduced_prototypes.append(reduced_prototype)
        reduced_targets.append(reduced_target)

        ''' Calculate ncc and store in lists'''
        ncc.append(normalized_cross_correlation(reduced_prototype, reduced_target))

    ''' Find index of poi which corresponds to highest ncc'''
    poi_index = ncc.index(max(ncc))    

    '''------------Poi positioned in middle of reduced_target-------------------'''
    #print(reduced_targets[0][reduced_size[0],reduced_size[1],reduced_size[2]])

    ''' Extract reduced data corresponding to highest ncc'''
    reduced_target = reduced_targets[poi_index]
    reduced_prototype = reduced_prototypes[poi_index]


    return reduced_target, reduced_prototype, poi_index


def cross_correlation(target_data, reduced_prototype, max_ncc_poi, reduced_size, extension):

    ''' Extend the reduced search space for cross correlation'''
    z_lower = max_ncc_poi[0]-extension-reduced_size[0]
    z_upper = max_ncc_poi[0]+extension+reduced_size[0]+1
    y_lower = max_ncc_poi[1]-extension-reduced_size[1]
    y_upper = max_ncc_poi[1]+extension+reduced_size[1]+1
    x_lower = max_ncc_poi[2]-extension-reduced_size[2]
    x_upper = max_ncc_poi[2]+extension+reduced_size[2]+1

    correlation_target = target_data[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]
    correlation_matrix = np.zeros((2*extension+1,2*extension+1,2*extension+1))
    center = np.array([extension, extension, extension])

    ''' Extract array shapes'''
    target_shape = correlation_target.shape
    prototype_shape = reduced_prototype.shape

    '''Hideous for-loop for doing cross-correlation'''
    for z in range(0,target_shape[0]-prototype_shape[0]+1):
        for y in range(0,target_shape[1]-prototype_shape[1]+1):
            for x in range(0,target_shape[2]-prototype_shape[2]+1):
                correlation_kernel = correlation_target[z:z+prototype_shape[0],y:y+prototype_shape[1],x:x+prototype_shape[2]]
                correlation_matrix[z,y,x] = normalized_cross_correlation(reduced_prototype, correlation_kernel)

    ''' Extract index of max ncc from correlation matrix'''
    max_ncc_index = unravel_index(correlation_matrix.argmax(), correlation_matrix.shape)

    ''' Calculate updated POI'''
    adjusted_poi = max_ncc_index - center + max_ncc_poi

    #print(correlation_matrix)
    #print(max_ncc_index)
    #print(adjusted_poi)

    return adjusted_poi