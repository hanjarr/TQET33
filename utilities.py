from amrafile import amrafile as af
from amracommon.analysis.registration import normalized_cross_correlation
from os import listdir
from os.path import isfile, join
import numpy as np


def init_poi(target_path, prototype_path, search_size, poi = 'T9'):

    ''' Get volumes from .amra format'''
    target = af.parse(target_path)
    prototypes = [join(prototype_path, f) for f in listdir(prototype_path) if isfile(join(prototype_path, f))]
    
    prototype_signals, prototype_pois  = [], []

    ''' Get poi positions from prototypes'''
    for index, prototype in enumerate(prototypes):
        prototype_signals.append(af.parse(prototype))
        prototype_pois.append(prototype_signals[index].get_poi(poi))
        
    poi_index, reduced_data = search_reduction(target, prototype_signals, prototype_pois, search_size)

    best_poi = prototype_pois[poi_index]

    print(best_poi)
    print(target.get_poi(poi))

    mean_poi = np.array(prototype_pois).mean(axis=0)

def search_reduction(target, prototype_signals, prototype_pois, search_size):
    voxel_sizes = target.voxel_size()
    size = np.round(search_size/np.array(voxel_sizes))
    size = size.astype(int)

    target_data = target.data
    prototype_data = [prototype.data for prototype in prototype_signals]

    reduced_prototypes, reduced_target, ncc = [], [], []

    for index, poi in enumerate(prototype_pois):
        prototype = prototype_data[index]

        z_lower = poi[0]-size[0]
        z_upper = poi[0]+size[0]
        y_lower = poi[1]-size[1]
        y_upper = poi[1]+size[1]
        x_lower = poi[2]-size[2]
        x_upper = poi[2]+size[2]

        reduced_prototypes.append(prototype[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper])
        reduced_target.append(target_data[z_lower:z_upper, y_lower:y_upper, x_lower:x_upper])

        ncc.append(normalized_cross_correlation(reduced_prototypes[index], reduced_target[index]))

    poi_index = ncc.index(max(ncc))

    reduced_data = reduced_target[poi_index]

    return poi_index, reduced_data




