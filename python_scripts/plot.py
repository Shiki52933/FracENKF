import numpy as np
import sys
import matplotlib
from matplotlib import pyplot as plt

plt.figure(figsize=[10,5])

for folder in [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]]:
    method = folder.split("/")[-2].split("_")[0]
    print("method: ", method)
    prefix1 = folder+"re"
    re_accumulated = None
    for i in range(1, 11):
        filename = prefix1+str(i)+".csv"
        one_re = np.loadtxt(filename)
        if re_accumulated is None:
            re_accumulated = one_re
        else:
            re_accumulated += one_re
    re_accumulated /= 10
    plt.subplot(1,2,1)
    plt.plot(range(len(re_accumulated)), re_accumulated)
    print(re_accumulated[-1])
plt.xlabel("number of iterations")
plt.legend(labels=['w=1','w=4','w=7','w=10','w=13'], loc='best')
plt.title("MRE")
# plt.savefig("./images/fenkf-mre.png")
# plt.show()

for folder in [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]]:
    prefix2 = folder+"rmse"
    rmse_accumulated = None
    for i in range(1, 11):
        filename = prefix2+str(i)+".csv"
        one_rmse = np.loadtxt(filename)
        if rmse_accumulated is None:
            rmse_accumulated = one_rmse
        else:
            rmse_accumulated += one_rmse
    # plt.plot(range(len(one_re)), one_rmse)
    # plt.title(filename)
    # plt.show()
    rmse_accumulated /= 10
    plt.subplot(1,2,2)
    plt.plot(range(len(rmse_accumulated)), rmse_accumulated)
    print(rmse_accumulated[-1])
plt.xlabel("number of iterations")
plt.legend(labels=['w=1','w=4','w=7','w=10','w=13'], loc='best')
plt.title("RMSE")

plt.savefig("./images/fenkf_w1-4-7-10-13.png")
plt.show()