import numpy as np
from matplotlib import pyplot as plt
from metric import *

sol = np.loadtxt("./data/lorenz63.csv")
# skewness = np.loadtxt("./data/skewness.csv")
# kurtosis = np.loadtxt("./data/kurtosis.csv")

fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.plot3D(sol[:,0], sol[:,1], sol[:,2], 'gray') 
plt.title("reference")
plt.show()
# plt.savefig("./images/lorenz63.png")
# plt.close()

analysis = np.loadtxt("./data/fgenkf_analysis.csv")

fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.plot3D(analysis[:,0], analysis[:,1], analysis[:,2], 'gray') 
plt.title("analysis")
plt.show()
# plt.savefig("./images/analysis.png")
# plt.close()

# rmse_re(analysis, sol, "./data/rmse.csv", "./data/re.csv")

error = analysis - sol
ab_error = np.max(np.abs(error), axis=-1)

fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.plot3D(error[:,0], error[:,1], error[:,2], 'gray') 
plt.title("error")
plt.show()

plt.plot(range(len(ab_error)), ab_error)
plt.title("absolutely error")
plt.show()
# plt.savefig("./images/ab_error.png")
# plt.close()

# plt.plot(range(len(skewness)), skewness)
# plt.title("skewness")
# plt.show()
# plt.savefig("./images/skewness.png")
# plt.close()
# print((skewness<1.04).sum() / len(skewness))

# plt.plot(range(len(kurtosis)), kurtosis)
# plt.title("kurtosis")
# plt.show()
# plt.savefig("./images/kurtosis.png")
# plt.close()

error = np.linalg.norm(error, axis=-1)/np.linalg.norm(sol, axis=-1)
error[error>1] = 1
plt.plot(range(len(error)), error)
plt.title("relatively error")
plt.show()
# plt.savefig("./images/re_error.png")
# plt.close()