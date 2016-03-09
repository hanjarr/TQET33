
from amrafile import amrafile as af
from amracommon.analysis.registration import normalized_cross_correlation
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def init_poi(target_path, prototype_path, search_size, poi = 'T9'):

    ''' Get target from .amra format and full paths to prototypes'''
    target = af.parse(target_path)
    prototype_paths = [join(prototype_path, f) for f in listdir(prototype_path) if isfile(join(prototype_path, f))]
    
    ''' Init empty lists for prototype signals and pois'''
    prototypes, prototype_pois = [], []

    ''' Get poi positions from prototypes'''
    for index, prototype in enumerate(prototype_paths):
        prototypes.append(af.parse(prototype))
        prototype_pois.append(prototypes[index].get_poi(poi))
    
    ''' Extract best poi according to ncc and the reduced data space'''    
    poi_index, poi_index_full, reduced_data = search_reduction(target, prototypes, prototype_pois, poi, search_size)

    ''' Number of prototypes used'''
    nbr_of_prototypes = len(prototypes)

    ''' Target poi ground truth'''
    target_poi = np.array(target.get_poi(poi))
    rep_target_poi = np.tile(target_poi, (nbr_of_prototypes,1))
    rep_voxel_sizes = np.tile(target.voxel_size(), (nbr_of_prototypes,1))

    ''' Diff between target poi ground truth and every deformed prototype poi in mm'''
    poi_diff = list(np.sum(abs(np.array(prototype_pois) - rep_target_poi)*rep_voxel_sizes, axis=1))

    ''' Differences sorted in ascending order'''
    sorted_diff = sorted(poi_diff)
    print(sorted_diff)

    sorted_index_reduced = sorted_diff.index(poi_diff[poi_index])
    sorted_index_full = sorted_diff.index(poi_diff[poi_index_full])

    best_poi = prototype_pois[poi_index]
    best_poi_full = prototype_pois[poi_index_full]

    ''' Best poi to choose from prototypes '''
    prototype_poi_index = poi_diff.index(min(poi_diff))
    sorted_index_best = sorted_diff.index(poi_diff[prototype_poi_index])

    print('\n'+'Target:' )
    print(target_poi)
    print('\n'+'Best ncc reduced:')
    print(best_poi)
    print(poi_diff[poi_index])
    print(sorted_index_reduced)
    print('\n'+'Best ncc full:')
    print(best_poi_full)
    print(poi_diff[poi_index_full])
    print(sorted_index_full)
    print('\n'+'Best prototype:' )
    print(prototype_pois[prototype_poi_index])
    print(poi_diff[prototype_poi_index])

    mean_poi = np.array(prototype_pois).mean(axis=0)


    return reduced_data

def plot_reduced(target_data, reduced_data, reduced_prototype, best_poi, best_poi_full, target_poi, reduced_size):

    ''' Show reduced data in target'''    
    plt.figure()
    plt.subplot(211)
    plt.imshow((reduced_data[:,reduced_size[1],:]), plt.get_cmap('gray'),origin='lower')

    ''' Show reduced data in prototype'''
    plt.subplot(212)
    plt.imshow((reduced_prototype[:,reduced_size[1],:]), plt.get_cmap('gray'),origin='lower')

    ''' Plot reduced area boxes around best poi when calculating full ncc and reduced. 
        Green marker is true poi'''
    plt.figure()
    currentAxis = plt.gca()
    plt.imshow((target_data[:, best_poi[1],:]), plt.get_cmap('gray'), origin='lower')
    plt.autoscale(False)
    plt.plot(target_poi[2], target_poi[0], marker='o', color='g')
    plt.plot(best_poi[2], best_poi[0], marker='o', color='b')
    plt.plot(best_poi_full[2], best_poi_full[0], marker='o', color='r')
    currentAxis.add_patch(Rectangle((best_poi[2]-reduced_size[2], best_poi[0]-reduced_size[0]), reduced_size[2]*2, reduced_size[0]*2, fill=None, edgecolor="blue"))
    currentAxis.add_patch(Rectangle((best_poi_full[2]-reduced_size[2], best_poi_full[0]-reduced_size[0]), reduced_size[2]*2, reduced_size[0]*2, fill=None, edgecolor="red"))

    plt.figure()
    currentAxis = plt.gca()
    plt.imshow((target_data[:,:,best_poi[2]]), plt.get_cmap('gray'), origin='lower')
    plt.autoscale(False)
    plt.plot(target_poi[1], target_poi[0], marker='o', color='g')
    plt.plot(best_poi[1], best_poi[0], marker='o', color='b')
    plt.plot(best_poi_full[1], best_poi_full[0], marker='o', color='r')
    currentAxis.add_patch(Rectangle((best_poi[1]-reduced_size[1], best_poi[0]-reduced_size[0]), reduced_size[1]*2, reduced_size[0]*2, fill=None, edgecolor="blue"))
    currentAxis.add_patch(Rectangle((best_poi_full[1]-reduced_size[1], best_poi_full[0]-reduced_size[0]), reduced_size[1]*2, reduced_size[0]*2, fill=None, edgecolor="red"))

    plt.figure()
    currentAxis = plt.gca()
    plt.imshow((target_data[best_poi[0],:,:]), plt.get_cmap('gray'), origin='lower')
    plt.autoscale(False)
    plt.plot(target_poi[2], target_poi[1], marker='o', color='g')
    plt.plot(best_poi[2], best_poi[1], marker='o', color='b')
    plt.plot(best_poi_full[2], best_poi_full[1], marker='o', color='r')
    currentAxis.add_patch(Rectangle((best_poi[2]-reduced_size[2], best_poi[1]-reduced_size[1]), reduced_size[2]*2, reduced_size[1]*2, fill=None, edgecolor="blue"))
    currentAxis.add_patch(Rectangle((best_poi_full[2]-reduced_size[2], best_poi_full[1]-reduced_size[1]), reduced_size[2]*2, reduced_size[2]*2, fill=None, edgecolor="red"))


    # plt.subplot(212)
    # #currentAxis = plt.gca()
    # plt.imshow((prototype_data[poi_index][:,best_poi[1],:]), plt.get_cmap('gray'), origin='lower')
    # #plt.plot(best_poi[2],best_poi[0], marker='o', color='b')
    # #currentAxis.add_patch(Rectangle((best_poi[2]-reduced_size[2], best_poi[0]-reduced_size[0]), reduced_size[2]*2, reduced_size[0]*2, fill=None, edgecolor="blue"))

    plt.show()

def search_reduction(target, prototypes, prototype_pois, poi, search_size):

    ''' Extract target voxel sizes and reduced voxel space'''
    voxel_sizes = target.voxel_size()
    reduced_size = np.round(search_size/np.array(voxel_sizes)).astype(int)

    ''' Get target and prototypes as numpy arrays'''
    target_data = target.data
    prototype_data = [prototype.data for prototype in prototypes]

    ''' Init empty lists for storing reduced prototypes, reduced target and ncc'''
    reduced_prototypes, reduced_targets, ncc, ncc_full = [], [], [], []

    ''' Target poi ground truth'''
    target_poi = np.array(target.get_poi(poi))

    ''' Calculate ncc between reduced target and reduced prototype space''' 
    for ind, poi in enumerate(prototype_pois):

        z_lower = poi[0]-reduced_size[0]
        z_upper = poi[0]+reduced_size[0]
        y_lower = poi[1]-reduced_size[1]
        y_upper = poi[1]+reduced_size[1]
        x_lower = poi[2]-reduced_size[2]
        x_upper = poi[2]+reduced_size[2]

        prototype = prototype_data[ind]

        ''' Extract reduced space from prototype and target'''
        reduced_prototype = prototype[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]
        reduced_target = target_data[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]

        ''' Append reduced numpy arrays to lists'''
        reduced_prototypes.append(reduced_prototype)
        reduced_targets.append(reduced_target)

        ''' Calculate ncc and store in lists'''
        ncc.append(normalized_cross_correlation(reduced_prototype, reduced_target))
        ncc_full.append(normalized_cross_correlation(prototype,target_data))

    print(reduced_targets[0].shape)

    ''' Find index of poi which corresponds to highest ncc'''
    poi_index = ncc.index(max(ncc))
    poi_index_full = ncc_full.index(max(ncc_full))
    
    ''' Extract reduced data corresponding to highest ncc'''
    reduced_data = reduced_targets[poi_index]
    reduced_prototype = reduced_prototypes[poi_index]

    ''' Best poi according to highest ncc measure'''
    best_poi = prototype_pois[poi_index]
    best_poi_full = prototype_pois[poi_index_full]

    plot_reduced(target_data, reduced_data, reduced_prototype, best_poi, best_poi_full, target_poi, reduced_size)


    return poi_index, poi_index_full, reduced_data






