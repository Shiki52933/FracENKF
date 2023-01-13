import numpy as np
from matplotlib import pyplot as plt

sol = np.loadtxt("./data/lorenz96.csv")
skewness = np.loadtxt("./data/skewness.csv")
kurtosis = np.loadtxt("./data/kurtosis.csv")

plt.plot(range(len(sol)), sol[:, -1])
plt.title("last component")
plt.show()
# plt.savefig("./images/lorenz96-1.png")
# plt.close()

analysis = np.loadtxt("./data/analysis.csv")
error = analysis - sol
ab_error = np.max(np.abs(error), axis=-1)

plt.plot(range(len(ab_error)), ab_error)
plt.title("absolutely error")
plt.show()
# plt.savefig("./images/ab_error.png")
# plt.close()

plt.plot(range(len(skewness)), skewness)
plt.title("skewness")
plt.show()
# plt.savefig("./images/skewness.png")
# plt.close()
# print((skewness<1.04).sum() / len(skewness))

plt.plot(range(len(kurtosis)), kurtosis)
plt.title("kurtosis")
plt.show()
# plt.savefig("./images/kurtosis.png")
# plt.close()

error = np.linalg.norm(error, axis=-1)/(np.linalg.norm(sol, axis=-1) + 1e-6)
error[ error > 1 ] = 1
#print(np.where(error>2)[0][0])
plt.plot(range(len(error)), error)
plt.title("relatively error")
plt.show()
# plt.savefig("./images/re_error.png")
# plt.close()