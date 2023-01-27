import numpy as np
from matplotlib import pyplot as plt

def rmse_re(analysis:np.ndarray, sol:np.ndarray, rmse_file:str, re_file:str):
    error = analysis - sol
    error_norm = np.linalg.norm(error, axis=1)

    cumsumed = np.cumsum(error_norm ** 2, axis=0)
    pre_rmse = cumsumed / np.arange(1, len(cumsumed)+1)
    plt.plot(range(len(pre_rmse)), np.sqrt(pre_rmse))
    plt.title("rmse")
    plt.show()
    np.savetxt(rmse_file, np.sqrt(pre_rmse))

    sol_norm = np.linalg.norm(sol, axis=1)
    re_norm = error_norm / sol_norm
    pre_re = np.cumsum(re_norm, axis=0)
    re = pre_re / np.arange(1, len(pre_re)+1)
    plt.plot(range(len(re)), re)
    plt.title("relative error")
    plt.show()
    np.savetxt(re_file, re)