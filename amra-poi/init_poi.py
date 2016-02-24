from amrafile import amrafile as af
from amracommon.analysis.registration import normalized_cross_correlation
from os import listdir
from os.path import isfile, join
import numpy as np

def init_poi(target_path, prototype_path):
    prototypes = [join(prototype_path, f) for f in listdir(prototype_path) if isfile(join(prototype_path, f))]
    target = af.parse(target_path)

    prototype_signals = []

    for prototype in prototypes:
        prototype_signals.append(af.parse(prototype))

        prototypes_t9, prototypes_lf, prototypes_rf = [], [], []

    for signal in prototype_signals:
        prototypes_t9.append(signal.get_poi('T9'))
        prototypes_lf.append(signal.get_poi('LeftFemur'))
        prototypes_rf.append(signal.get_poi('RightFemur'))

    target_data = target.data
    prototypes_data = [prototype.data for prototype in prototype_signals]
    ncc = [normalized_cross_correlation(prototype, target_data) for prototype in prototypes_data]

    max_index = ncc.index(max(ncc))
    print(max_index)

    prototypes_t9 = np.array(prototypes_t9)
    prototypes_lf = np.array(prototypes_lf)
    prototypes_rf = np.array(prototypes_rf)

    print(prototypes_rf[max_index])
    print(target.get_poi('RightFemur'))

    mean_t9 = prototypes_t9.mean(axis=0)
    mean_lf = prototypes_lf.mean(axis=0)
    mean_rf = prototypes_rf.mean(axis=0)
