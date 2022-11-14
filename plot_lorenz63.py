import numpy as np
from matplotlib import pyplot as plt

sol = np.loadtxt("lorenz63.csv")

fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.plot3D(sol[:,0], sol[:,1], sol[:,2], 'gray') 
plt.savefig("lorenz63.png")