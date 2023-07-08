# this is a script to plot the shallow water equations
# read ./data/shallowwater/sol*.csv
# plot the solution at each time step
# save the plots in ./images/shallowwater/

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv

# read the data
def read_data(filename, cols, rows):
    data = np.loadtxt(filename)
    data = data.reshape((3, cols, rows))
    return data

# plot the solution
def plot_solution(data, filename):
    u = data[0]
    v = data[1]
    h = data[2]
    # u,v,h are matrix, plot them in a figure
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(u)
    plt.title('u')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(v)
    plt.title('v')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(h)
    plt.title('h')
    plt.colorbar()
    plt.savefig(filename)
    plt.show()
    plt.close()


# main function
def main():
    # read the data
    data_dir = './data/shallowwater/'
    data_files = os.listdir(data_dir)
    data_files.sort()
    for data_file in data_files:
        data = read_data(data_dir + data_file, 513, 513)
        # plot the solution
        image_dir = './images/shallowwater/'
        image_file = image_dir + data_file[:-4] + '.png'
        plot_solution(data, image_file)
    
if __name__ == '__main__':
    main()