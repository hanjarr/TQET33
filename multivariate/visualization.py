import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

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

    reg_hist = np.histogram(reg_error, range(np.amax(reg_error).astype(int)))
    ncc_hist = np.histogram(ncc_error, range(np.amax(ncc_error).astype(int)))

    reg_voxel_hist = np.histogram(reg_voxel_error, range(np.amax(reg_voxel_error).astype(int)))
    ncc_voxel_hist = np.histogram(ncc_voxel_error, range(np.amax(ncc_voxel_error).astype(int)))

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