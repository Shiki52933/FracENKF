# read from ./data/max_error.bin and plot the error
# read from ./data/rel_error.bin and plot the error

import numpy as np
import sys
from matplotlib import pyplot as plt



# plot reference solution
sol = np.fromfile("./data/swe/sol0.000000.bin", dtype=np.float64)
sol.shape = [int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])]
sol = sol[:,:,2].transpose()
plt.figure(figsize=(10,10))
plt.subplot(3,3,1)
plt.imshow(sol, cmap='jet', interpolation='none', aspect='auto')
plt.title("reference")
plt.colorbar()
# plt.ion()
# plt.show()
# plt.savefig('./images/swe/ref.png')
# plt.pause(1)
# plt.clf()


swe_init = np.fromfile('./data/swe_init.bin', dtype=np.float64)
one_len = int(sys.argv[1])*int(sys.argv[2])*int(sys.argv[3])
swe_init.shape = [one_len * int(sys.argv[4])]
for i in range(int(sys.argv[4])):
    one_en = swe_init[one_len*i : one_len*(i+1)]
    one_en = one_en.reshape([int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])])
    if i+2 > 9:
        continue
    plt.subplot(3,3,i+2)
    plt.imshow(one_en[:,:,2].transpose(), cmap='jet', interpolation='none', aspect='auto')
    plt.title("ensemble{}".format(i))
    plt.colorbar()
    # plt.savefig('./images/swe/ensemble{}.png'.format(i))
    print("plotting: ", i)
    # plt.pause(.1)
    # plt.clf()
# plt.colorbar()
plt.tight_layout()
plt.show()
# plt.ioff()
# plt.close()

max_error = np.fromfile('./data/max_error.bin', dtype=np.float64)
plt.clf()
plt.plot(max_error)
plt.grid(axis='both', which='major')
plt.title('max error')
plt.show()

rel_error = np.fromfile('./data/rel_error.bin', dtype=np.float64)
plt.plot(rel_error)
plt.grid(axis='both', which='major')
plt.title('relative error')
plt.show()
