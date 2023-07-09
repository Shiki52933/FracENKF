# this is a script to plot the shallow water equations
# read ./data/shallowwater/sol*.csv
# plot the solution at each time step
# save the plots in ./images/shallowwater/

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# read the data
def read_data(filename, cols, rows):
    data = np.fromfile(filename, dtype=np.float64)
    data.shape = [3, cols, rows]
    return data

first_time = True
# plot the solution
def plot_solution(data, dof):
    global first_time
    # u,v,h are matrix, plot them in a figure
    plt.imshow(data[dof], cmap='jet', interpolation='nearest')
    if first_time:
        print("plotting")
        plt.ion()
        plt.show()
    # plt.savefig(filename)
    sys.stdout.flush()
    print('.')
    plt.pause(0.001)
    first_time = False


# main function
def main():
    # read the data
    data_dir = './data/shallowwater/'
    data_files = os.listdir(data_dir)
    data_files.sort()
    for data_file in data_files:
        data = read_data(data_dir + data_file, 129, 129)
        # plot the solution
        # image_dir = './images/shallowwater/'
        # image_file = image_dir + data_file[:-4] + '.png'
        plot_solution(data, int(sys.argv[1]))
    
if __name__ == '__main__':
    main()