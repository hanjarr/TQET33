
from amrafile import amrafile as af
from amracommon.analysis.registration import normalized_cross_correlation
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt


def init_poi(target_path, prototype_path, search_size, poi = 'T9'):

    ''' Get volumes from .amra format'''
    target = af.parse(target_path)
    prototypes = [join(prototype_path, f) for f in listdir(prototype_path) if isfile(join(prototype_path, f))]
    
    prototype_signals, prototype_pois = [], []

    ''' Get poi positions from prototypes'''
    for index, prototype in enumerate(prototypes):
        prototype_signals.append(af.parse(prototype))
        prototype_pois.append(prototype_signals[index].get_poi(poi))
    
    ''' Extract best poi according to ncc and the reduced data space'''    
    best_poi, reduced_data = search_reduction(target, prototype_signals, prototype_pois, search_size)

    ''' True poi'''
    target_poi = np.array(target.get_poi(poi))
    rep_poi = np.tile(target_poi, (len(prototype_pois),1))
    rep_voxel_sizes = np.tile(target.voxel_size(),(len(prototype_pois),1))

    ''' Diff for every deformed prototype poi in milimeters'''
    poi_diff = list(np.sum(abs(np.array(prototype_pois) - rep_poi)*rep_voxel_sizes, axis=1))

    print(poi_diff)

    ''' Best poi to choose from prototypes '''
    prototype_poi_index = poi_diff.index(min(poi_diff))

    print('\n'+'Target:' )
    print(target_poi)
    print('\n'+'Best ncc:')
    print(best_poi)
    print('\n'+'Best prototype:' )
    print(prototype_pois[prototype_poi_index])

    mean_poi = np.array(prototype_pois).mean(axis=0)

    return reduced_data

def search_reduction(target, prototype_signals, prototype_pois, search_size):
    voxel_sizes = target.voxel_size()
    reduced_size = np.round(search_size/np.array(voxel_sizes))
    reduced_size = reduced_size.astype(int)

    target_data = target.data
    prototype_data = [prototype.data for prototype in prototype_signals]

    reduced_prototypes, reduced_targets, ncc = [], [], []

    for ind, poi in enumerate(prototype_pois):
        prototype = prototype_data[ind]

        z_lower = poi[0]-reduced_size[0]-1
        z_upper = poi[0]+reduced_size[0]
        y_lower = poi[1]-reduced_size[1]-1
        y_upper = poi[1]+reduced_size[1]
        x_lower = poi[2]-reduced_size[2]-1
        x_upper = poi[2]+reduced_size[2]

        reduced_prototype = prototype[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]
        reduced_target = target_data[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper]

        reduced_prototypes.append(reduced_prototype)
        reduced_targets.append(reduced_target)

        ncc.append(normalized_cross_correlation(reduced_prototype, reduced_target))

    poi_index = ncc.index(max(ncc))
    
    reduced_data = reduced_targets[poi_index]
    reduced_prototype = reduced_prototypes[poi_index]

    best_poi = prototype_pois[poi_index]

    plt.figure(1)
    plt.colorbar
    plt.subplot(211)
    plt.imshow(np.flipud(reduced_data[:,np.round(reduced_size[2]/2),:]), plt.get_cmap('gray'))

    plt.subplot(212)
    plt.imshow(np.flipud(reduced_prototype[:,np.round(reduced_size[2]/2),:]), plt.get_cmap('gray'))

    plt.show()

    # plt.figure(1)
    # plt.colorbar
    # plt.subplot(211)
    # plt.imshow(np.flipud(target_data[:,best_poi[1],:]), plt.get_cmap('gray'))

    # plt.subplot(212)
    # plt.imshow(np.flipud(prototype_data[poi_index][:,best_poi[1],:]), plt.get_cmap('gray'))

    # plt.show()


    return best_poi, reduced_data




