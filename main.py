from utilities import init_poi
from amrafile import amrafile as af
import numpy as np


def main():
    
    target = '0015B'
    directory = '0030/'
    target_path = '/moria/data/DB/'+directory+target+'/wholebody_normalized_water_1_'+target+'.amra'
    prototype_path = '/home/hannes/DB/'+directory+target+'/prototypes'

    search_size = np.array([40,40,40])

    init_poi(target_path,prototype_path, search_size, 'LeftFemur')

if __name__ == "__main__":
    main()