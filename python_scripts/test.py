import numpy as np
from matplotlib import pyplot as plt
import sys


errors = np.loadtxt('./data/lorenz96.csv')
errors[errors > 1] = 1
plt.plot(errors)
plt.legend(['enkf1', 'enkf2', 'genkf1', 'genkf2', 'ggenkf'])
plt.show()