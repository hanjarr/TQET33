from utilities import init_poi
import numpy as np

def main():

    target_path = '/moria/data/DB/0030/0015A/wholebody_normalized_water_1_0015A.amra'
    prototype_path = '/home/hannes/DB/0030/0015A/prototypes'

    search_size = np.array([30,30,30])

    init_poi(target_path,prototype_path, search_size, 'T9')

if __name__ == "__main__":
    main()