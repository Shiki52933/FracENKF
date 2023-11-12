import numpy as np
from matplotlib import pyplot as plt

from python_scripts.metric import rmse_re

sol = np.loadtxt("./data/lorenz96.csv")
# skewness = np.loadtxt("./data/skewness.csv")
# kurtosis = np.loadtxt("./data/kurtosis.csv")

# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# ax1.plot3D(sol[:,0], sol[:,1], sol[:,2], 'gray') 
# plt.title("reference 3 components")
# plt.show()

analysis = np.loadtxt("./data/analysis.csv")

# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# ax1.plot3D(analysis[:,0], analysis[:,1], analysis[:,2], 'gray') 
# plt.title("analysis 3 components")
# plt.show()

rmse_re(analysis, sol, "./data/rmse.csv", "./data/re.csv")

error = analysis - sol
ab_error = np.linalg.norm(error, axis=-1)

# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# ax1.plot3D(error[:,0], error[:,1], error[:,2], 'gray') 
# plt.title("error")
# plt.show()

# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# ax1.plot3D(analysis[:,0], analysis[:,1], analysis[:,2], 'gray') 
# plt.title("analysis 3 components")
# plt.show()

plt.plot(range(len(ab_error)), ab_error)
plt.title("absolutely error")
plt.show()


# plt.plot(range(len(skewness)), skewness)
# plt.title("skewness")
# plt.show()

# plt.plot(range(len(kurtosis)), kurtosis)
# plt.title("kurtosis")
# plt.show()

# error = np.linalg.norm(error, axis=-1)/(np.linalg.norm(sol, axis=-1) + 1e-6)
# error[ error > 1 ] = 1

# plt.plot(range(len(error)), error)
# plt.title("relatively error")
# plt.show()
