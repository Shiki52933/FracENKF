import numpy as np
from matplotlib import pyplot as plt

sol = np.loadtxt("lorenz63.csv")

fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.plot3D(sol[:,0], sol[:,1], sol[:,2], 'gray') 
plt.show()
# plt.savefig("lorenz63.png")

analysis = np.loadtxt("analysis.csv")

fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.plot3D(analysis[:,0], analysis[:,1], analysis[:,2], 'gray') 
plt.show()
# plt.savefig("lorenz63.png")

error = analysis - sol
error = np.max(np.abs(error), axis=-1)
plt.plot(range(len(error)), error)
plt.show()
#plt.savefig("error.png")