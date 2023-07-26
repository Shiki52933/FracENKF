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
def read_data(filename, cols, rows, dof):
    data = np.fromfile(filename, dtype=np.float64)
    data.shape = [cols, rows, int(sys.argv[3])]
    return data[:,:,dof].transpose()


# plot the solution
def plot_solution(datas, t, dof):
    first_time = True
    critical_times = [0, 350, 510, 650]
    # u,v,h are matrix, plot them in a figure
    for i, data in enumerate(datas):
        plt.imshow(data, cmap='jet', interpolation='none')
        plt.colorbar()
        if first_time:
            print("plotting")
            plt.ion()
            plt.show()
        # plt.savefig(filename)
        sys.stdout.flush()
        print("time = ", t[i][3:-4])
        if(int(float(t[i][3:-4])) in critical_times):
            plt.pause(5)
        plt.pause(0.001)
        plt.clf()
        first_time = False


# main function
def main():
    # read the data 
    data_dir = sys.argv[5]
    data_files = os.listdir(data_dir)
    data_files.sort(key=lambda x:float(x[3:-4]))
    data_files = data_files[0:500]
    # data_files.sort()
    
    datas = []
    for data_file in data_files:
        data = read_data(data_dir + data_file, int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[4]))
        datas.append(data)
        
    # plot the solution    
    plot_solution(datas, data_files, int(sys.argv[4]))
    plt.close()
    
if __name__ == '__main__':
    main()