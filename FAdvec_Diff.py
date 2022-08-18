import math
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from enKF_param import ObserveErrorMatrices, ObserveOperator


def StochasticEnsembleKF(num_ensemble: int, init_ave: np.ndarray, init_uncertainty: np.ndarray, observations: list,
                         iters: int, Ob_Errors: ObserveErrorMatrices, Ob_Ops: ObserveOperator, Models, sys_vars):
    # initializing
    analysis_results = []
    ensemble = np.random.multivariate_normal(init_ave, init_uncertainty, size=num_ensemble).T

    for idx in range(0, iters):
        if observations[idx] is not None:
            # generate perturbed observations
            perturbations = np.random.multivariate_normal(np.zeros(observations[idx].shape[0]),
                                                        Ob_Errors.get_next(), size=num_ensemble).T
            after_perturb = observations[idx] + perturbations

            # compute
            ensemble_mean = np.mean(ensemble, axis=1, keepdims=True)
            perturb_mean = np.mean(perturbations, axis=1, keepdims=True)

            # 为了避免低效的循环，要求观测算子能够直接处理一整个集合
            y_f = Ob_Ops.apply(ensemble)

            x_f = (ensemble - ensemble_mean) / np.sqrt(num_ensemble - 1)
            y_mean = np.mean(y_f, axis=1, keepdims=True)

            auxiliary = after_perturb - y_f
            y_f = (y_f - perturbations - y_mean + perturb_mean) / np.sqrt(num_ensemble - 1)

            # compute gain
            gain = x_f.dot(y_f.T).dot(np.linalg.inv(y_f.dot(y_f.T)))

            # update ensemble
            ensemble_analysis = ensemble + gain.dot(auxiliary)

        else:
            ensemble_analysis = ensemble

        # test
        ensemble_mean = np.mean(ensemble_analysis, axis=1, keepdims=True)
        analysis_results.append(ensemble_mean)

        # 为了避免低效的循环，要求模型算子能够直接处理一整个集合
        if idx != iters-1:
            ensemble = Models(ensemble_analysis, idx, sys_vars[idx])

    return analysis_results


class FracAdvecDiffDiri:
    def __init__(self, L, R, T, alpha, beta, gamma, LBd, RBd, v, d, f, xNum, tNum):
        self.LBd = LBd
        self.RBd = RBd

        # 计算一些额外的东西备用
        self.b_alpha = B_alpha(alpha, tNum + 1)
        self.x_mesh = np.linspace(L, R, xNum)
        self.t_mesh = np.linspace(0, T, tNum)
        self.v_mesh = v(self.t_mesh, self.x_mesh) * (T / tNum) ** alpha * math.gamma(2 - alpha) / ((R - L) / xNum) ** beta
        self.d_mesh = d(self.t_mesh, self.x_mesh) * (T / tNum) ** alpha * math.gamma(2 - alpha) / ((R - L) / xNum) ** gamma
        self.f_mesh = f(np.expand_dims(self.t_mesh, axis=1), np.expand_dims(self.x_mesh, axis=0)) * (T / tNum) ** alpha * math.gamma(
                        2 - alpha)
        self.w_beta = coefficient(xNum + 1, [beta]).squeeze()
        self.w_gamma = coefficient(xNum + 2, [gamma]).squeeze()

    def getModel(self):
        def FracAdvecDiffModel(input:np.ndarray, idx, sys_var):
            """遵循的原则是：时间上在后的在下"""
            mesh_size = sys_var.shape[0]

            # 计算系数矩阵
            A = np.zeros([mesh_size, mesh_size], dtype=np.float64)
            A[0, 0] = 1
            A[-1, -1] = 1
            for i in range(1, mesh_size - 1):
                A[i, i] += 1
                A[i, 0:i + 1] += self.v_mesh[idx + 1, i] * self.w_beta[i::-1]
                A[i, 0:i + 2] -= self.d_mesh[idx + 1, i] * self.w_gamma[i + 1::-1]

            # 计算右端项
            rhs = np.zeros([mesh_size, input.shape[1]])
            rhs[0] = self.LBd(self.t_mesh[idx + 1])
            rhs[-1] = self.RBd(self.t_mesh[idx + 1])

            for j in range(0, input.shape[1]):
                sol = input[:, j].reshape([input.shape[0] // mesh_size, mesh_size])

                for i in range(1, mesh_size - 1):
                    rhs[i,j] += np.sum((self.b_alpha[0:idx] - self.b_alpha[1:idx+1]) * sol[idx:0:-1, i])
                    rhs[i,j] += self.b_alpha[idx] * sol[0, i] + self.f_mesh[idx+1, i]

            new_sol = np.linalg.inv(A).dot(rhs)
            output = np.concatenate([input, new_sol], axis=0)
            return output

        return FracAdvecDiffModel


class FracAdvecDiffInverse:
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.xNum = 501
        self.tNum = 2501
        self.L = 0
        self.R = 1
        self.T = 5
        self.stNum = 100
        self.sxNum = 20

        # self.f = lambda t,x: np.sin(np.pi * x.reshape([1, x.shape[-1]]) * t.reshape([t.shape[0], 1])) + \
        #                     x.reshape([1, x.shape[-1]]) ** 2 * t.reshape([t.shape[0], 1])
        # self.f_scala = lambda t,x: np.sin(np.pi * x * t) + x ** 2 * t

        self.f = lambda t,x: self.f_scala( t.reshape([t.shape[0],1]), x.reshape([1,x.shape[-1]]) )
        self.f_scala = lambda t,x: x**2 + 2*x*t + 2*x**3*t + 6*x*x*t -2*t - 4*t*t
        self.real_a_x = lambda x : x*x
        self.real_b_x = lambda x : 3*x
        self.real_c_x = lambda x : 1
        self.real_d_x = lambda x : 2

        self.LBd = lambda t: 0
        self.RBd = lambda t: t
        self.init = lambda x: 0

        self.w_beta = coefficient(self.sxNum + 1, [beta]).squeeze()
        self.w_gamma = coefficient(self.sxNum + 2, [gamma]).squeeze()

        self.b_alpha = B_alpha(alpha, self.stNum + 2)


    def generate_solution(self):
        L = self.L
        R = self.R
        T = self.T
        xNum = self.xNum
        tNum = self.tNum

        def v(t, x):
            t_copy = t.reshape([t.shape[0], 1])
            x_copy = x.reshape([1, x.shape[-1]])
            return self.v_init(t_copy, x_copy, self.real_a_x(x_copy), self.real_b_x(x_copy))

        def d(t, x):
            t_copy = t.reshape([t.shape[0], 1])
            x_copy = x.reshape([1, x.shape[-1]])
            return self.d_init(t_copy, x_copy, self.real_c_x(x_copy), self.real_d_x(x_copy))

        sol = FracAdvecDiffDirichletSolver(L, R, T, self.alpha, self.beta, self.gamma, v, d, self.f,
                                            self.LBd, self.RBd, self.init, xNum, tNum)

        # 生成观测数据
        ob_origin = sol[::125, ::125].copy()
        ob_origin += np.random.normal(0, 0.5, size=ob_origin.shape)

        return sol, ob_origin

    @staticmethod
    def v_init(t, x, a_x, b_x):
        return (1 + a_x + b_x ) * np.ones_like(t) * np.ones_like(x)

    @staticmethod
    def d_init(t, x, c_x, d_x):
        return c_x + d_x * t * np.ones_like(x)      # c_x + d_x * x * t

    @staticmethod
    def get_ob_op():
        def ob_op(input):
            return input[-105:-84:5]
        return ob_op

    def get_model_op(self):
        def FracAdvecDiffInverseModel(input:np.ndarray, idx, sys_var):
            """遵循的原则是：时间上在后的在下,未知参数放最下面，这里每个网格点有4个相应的未知参数"""
            mesh_size = sys_var.shape[0] // 5
            data = input[:-4*mesh_size]
            new_sol = np.ndarray([mesh_size, input.shape[1]])
            param = input[-4*mesh_size:]

            # 网格是需要的
            x_mesh = np.linspace(self.L, self.R, mesh_size)

            for j in range(0, input.shape[1]):
                # 计算辅助项
                ax = param[:mesh_size, j]
                bx = param[mesh_size:2*mesh_size, j]
                cx = param[2*mesh_size:3*mesh_size, j]
                dx = param[3*mesh_size:, j]

                # 计算格点上需要的v、d离散后的值，特定于每一个集合成员
                v_mesh = self.v_init(idx*self.T/self.stNum, x_mesh, ax, bx) * \
                         (self.T / self.stNum) ** self.alpha * math.gamma(2 - self.alpha) / \
                         ((self.R - self.L) / self.sxNum) ** self.beta

                d_mesh = self.d_init(idx*self.T/self.stNum, x_mesh, cx, dx) * \
                         (self.T / self.stNum) ** self.alpha * math.gamma(2 - self.alpha) / \
                         ((self.R - self.L) / self.sxNum) ** self.gamma

                # 计算系数矩阵
                A = np.zeros([mesh_size, mesh_size], dtype=np.float64)
                A[0, 0] = 1
                A[-1, -1] = 1
                for i in range(1, mesh_size - 1):
                    A[i, i] += 1
                    A[i, 0:i + 1] += v_mesh[i] * self.w_beta[i::-1]
                    A[i, 0:i + 2] -= d_mesh[i] * self.w_gamma[i + 1::-1]

                # 计算右端项
                rhs = np.zeros([mesh_size, 1])
                rhs[0] = self.LBd(idx*self.T/self.stNum)
                rhs[-1] = self.RBd(idx*self.T/self.stNum)

                sol = data[:, j].reshape([data.shape[0] // mesh_size, mesh_size])

                for i in range(1, mesh_size - 1):
                    rhs[i] += np.sum((self.b_alpha[0:idx] - self.b_alpha[1:idx+1]) * sol[idx:0:-1, i])
                    rhs[i] += self.b_alpha[idx] * sol[0, i] + self.f_scala(idx*self.T/self.stNum, x_mesh[i]) * \
                                (self.T / self.stNum) ** self.alpha * math.gamma(2 - self.alpha)

                new_sol[:,j] = np.linalg.inv(A).dot(rhs).squeeze()

            # new_sol += np.random.multivariate_normal(np.zeros(mesh_size), sys_var, size=[input.shape[1]]).T
            output = np.concatenate([data, new_sol, param], axis=0)
            output[-5*mesh_size:] += np.random.multivariate_normal(np.zeros(sys_var.shape[0]), sys_var, size=[input.shape[1]]).T
            return output

        return FracAdvecDiffInverseModel



def B_alpha(alpha, N):
    j = np.ndarray([N+2])
    j[0] = 0
    for i in range(1, j.shape[0]):
        j[i] = i ** (1-alpha)
    return j[1:] - j[:-1]


def coefficient(len, orders):
    # 计算系数
    orders = np.array(orders)
    bino = np.ndarray([len, orders.shape[0]])
    bino[0] = 1
    for i in range(1, len):
        bino[i] = (1 - (1 + orders) / i) * bino[i - 1]
    return bino


def FracAdvecDiffDirichletSolver(L, R, T, alpha, beta, gamma, v, d, f, LBd, RBd, init, xNum, tNum):
    sol = np.ndarray([tNum, xNum], dtype=np.float64)
    sol[0] = init(np.linspace(L, R, xNum))

    # 预备工作
    b_alpha = B_alpha(alpha, tNum+1)
    x_mesh = np.linspace(L, R, xNum)
    t_mesh = np.linspace(0, T, tNum)
    v_mesh = v(t_mesh, x_mesh) * (T/tNum) ** alpha * math.gamma(2-alpha) / ((R-L)/xNum) ** beta
    d_mesh = d(t_mesh, x_mesh) * (T/tNum) ** alpha * math.gamma(2-alpha) / ((R-L)/xNum) ** gamma
    f_mesh = f(np.expand_dims(t_mesh,axis=1), np.expand_dims(x_mesh,axis=0)) * (T/tNum) ** alpha * math.gamma(2 - alpha)

    w_beta = coefficient(xNum+1, [beta]).squeeze()
    w_gamma = coefficient(xNum+2, [gamma]).squeeze()

    # 计算系数矩阵
    for k in range(1, tNum):
        A = np.zeros([xNum, xNum], dtype=np.float64)
        A[0, 0] = 1
        A[-1, -1] = 1
        for i in range(1, xNum-1):
            A[i,i] += 1
            A[i,0:i+1] += v_mesh[k,i] * w_beta[i::-1]
            A[i,0:i+2] -= d_mesh[k,i] * w_gamma[i+1::-1]

        rhs = np.zeros([xNum, 1])
        rhs[0] = LBd(t_mesh[k])
        rhs[-1] = RBd(t_mesh[k])
        for i in range(1, xNum-1):
            rhs[i] += np.sum( ( b_alpha[0:k-1] - b_alpha[1:k] ) * sol[k-1:0:-1,i] )
            rhs[i] += b_alpha[k-1] * sol[0,i] + f_mesh[k,i]

        sol[k] = np.linalg.inv(A).dot(rhs).squeeze()

    return sol


def FracAdvecDiffDirichletSolverTest():
    def v(t,x):
        return np.ones([t.shape[0], x.shape[0]])

    def d(t,x):
        return np.ones([t.shape[0], x.shape[0]])

    def f(t,x):
        return x*x + 2*x*t - 2*t

    def LBd(t):
        return 0

    def RBd(t):
        return t

    def init(x):
        return np.zeros_like(x)

    L = 0
    R = 1
    T = 1
    alpha = 1
    beta = 1
    gamma = 2
    xNum = 501
    tNum = 501

    sol = FracAdvecDiffDirichletSolver(L,R,T,alpha,beta,gamma,v,d,f,LBd,RBd,init,xNum,tNum)

    # ref1 = sol[0::25, 0::25]

    x = np.expand_dims(np.linspace(L, R, xNum), axis=0)
    t = np.expand_dims(np.linspace(0, T, tNum), axis=1)
    ref = x*x*t
    print(np.max(np.abs(ref-sol)))


def FracAdvecDiffENKFTest():
    def v(t,x):
        return np.ones([t.shape[0], x.shape[0]])

    def d(t,x):
        return np.ones([t.shape[0], x.shape[0]])

    def f(t,x):
        return x*x + 2*x*t - 2*t

    def LBd(t):
        return 0

    def RBd(t):
        return t

    def init(x):
        return np.zeros_like(x)

    def ob_operator(input):
        return input[-21::5]

    L = 0
    R = 1
    T = 1
    alpha = 0.9
    beta = 0.9
    gamma = 1.8
    xNum = 501
    tNum = 501

    sol = FracAdvecDiffDirichletSolver(L, R, T, alpha, beta, gamma, v, d, f, LBd, RBd, init, xNum, tNum)

    # 生成观测数据
    ob_origin = sol[::125,::125].copy()
    ob_origin += np.random.normal(0,0.5,size=ob_origin.shape)

    # 实例化模型类
    model = FracAdvecDiffDiri(L, R, T, alpha, beta, gamma, LBd, RBd, v, d, f, 21, 21)
    model_op = model.getModel()

    # 生成enkf需要的变量
    ob_list = []
    for i in range(4):
        ob_list += [ob_origin[i].reshape([ob_origin.shape[1], 1])] + [None] * 4
    ob_list += [ob_origin[-1].reshape([ob_origin.shape[1], 1])]

    # 生成观测误差矩阵
    iters = len(ob_list)
    ob_err = np.identity(ob_origin.shape[1]) * 0.5
    ob_errs = [ob_err] * len(ob_list)
    ob_errors = ObserveErrorMatrices(ob_errs)

    # 生成观测算子
    ob_op = ObserveOperator(ob_operator)

    # 系统误差
    sys_var = np.identity(21) * 0.01
    sys_vars = [sys_var] * iters

    # 初始值
    init_ave = np.zeros([21])
    init_var = 10 * np.identity(21)

    # enkf
    results = StochasticEnsembleKF(20,init_ave,init_var,ob_list,iters,ob_errors,ob_op,model_op,sys_vars)

    # 评价结果
    final = results[-1]
    final = final.reshape([final.shape[0]//21,21])

    for i in range(len(results)):
        results[i] = results[i][-21:].squeeze()
    results = np.array(results)

    ref1 = sol[0::25,0::25]
    # x = np.expand_dims(np.linspace(L, R, 501), axis=0)
    # t = np.expand_dims(np.linspace(0, T, 501), axis=1)
    # ref = x * x * t
    # print(np.max(np.abs(sol-ref)))
    # print(results.shape, final.shape, ref1.shape)

    abs_err = np.max(np.abs(results - ref1), axis=1)
    plt.plot(range(abs_err.shape[0]), abs_err)
    plt.show()
    abs_err = np.max(np.abs(final - ref1), axis=1)
    plt.plot(range(abs_err.shape[0]), abs_err)
    plt.show()


def FracAdvecDiffInverseENKFTest():
    example = FracAdvecDiffInverse(0.8,0.8,1.5)
    model_op = example.get_model_op()
    # sol, ob_origin = example.generate_solution()
    # np.savez("./data/分数对流扩散/simplest_0_1_0_5.npz", real=sol, ob=ob_origin)
    read = np.load("./data/分数对流扩散/simplest_0_1_0_5.npz")
    sol, ob_origin = read['real'], read['ob']

    # 画真解
    x_mesh = np.linspace(example.L, example.R, sol.shape[1])
    t_mesh = np.linspace(0, example.T, sol.shape[0])
    t_mesh, x_mesh = np.meshgrid(t_mesh, x_mesh, indexing='ij')
    fig = plt.figure()
    ax = Axes3D(fig)
    # print(x_mesh.shape, t_mesh.shape, sol.shape)
    ax.plot_surface(t_mesh, x_mesh, sol)
    plt.show()

    # 生成enkf需要的变量
    ob_list = []
    for i in range(ob_origin.shape[0]-1):
        ob_list += [ob_origin[i].reshape([ob_origin.shape[1], 1])] + [None] * 4
    ob_list += [ob_origin[-1].reshape([ob_origin.shape[1], 1])]

    # 生成观测误差矩阵
    iters = len(ob_list)
    ob_err = np.identity(ob_origin.shape[1]) * 0.5
    ob_errs = [ob_err] * len(ob_list)
    ob_errors = ObserveErrorMatrices(ob_errs)

    # 生成观测算子
    ob_op = ObserveOperator(example.get_ob_op())

    # 系统误差
    sys_var = np.identity(21*5) * 0.01
    sys_vars = [sys_var] * iters

    # 初始值
    real_a_x = np.linspace(example.L, example.R, example.sxNum+1) ** 2
    real_b_x = np.linspace(example.L, example.R, example.sxNum+1) * 3
    real_c_x = np.ones_like(real_a_x)
    real_d_x = 2 * real_c_x
    init_ave = np.concatenate([np.zeros_like(real_a_x), real_a_x, real_b_x, real_c_x, real_d_x])
    # init_ave += np.random.multivariate_normal(np.zeros_like(init_ave), 0.15*np.identity(init_ave.shape[-1]))

    # init_ave = np.zeros([21*5])
    init_var = np.identity(21*5)
    init_var[:21] *= 5
    init_var[21:] *= 1

    # enkf
    N = 20
    fianl_sum = 0
    results_sum = 0
    for idx in range(1, N+1):
        ob_errors = ObserveErrorMatrices(ob_errs)
        results = StochasticEnsembleKF(100,init_ave,init_var,ob_list,iters,ob_errors,ob_op,model_op,sys_vars)

        # 评价结果
        final = results[-1]
        final = final.reshape([final.shape[0]//21,21])
        fianl_sum = fianl_sum + final

        for i in range(len(results)):
            results[i] = results[i][-21*5:-21*4].squeeze()
        results = np.array(results)
        results_sum = results_sum + results

    final = fianl_sum / N
    results = results_sum / N

    ref1 = sol[0::25,0::25]
    # x = np.expand_dims(np.linspace(L, R, 501), axis=0)
    # t = np.expand_dims(np.linspace(0, T, 501), axis=1)
    # ref = x * x * t
    # print(np.max(np.abs(sol-ref)))
    # print(results.shape, final.shape, ref1.shape)

    # 画图
    abs_err = np.max(np.abs(results - ref1), axis=1)
    plt.plot(range(abs_err.shape[0]), abs_err)
    plt.show()
    abs_err = np.max(np.abs(final[:-4] - ref1), axis=1)
    plt.plot(range(abs_err.shape[0]), abs_err)
    plt.show()

    plt.plot(np.linspace(example.L, example.R, example.sxNum+1), final[-4])
    plt.show()
    plt.plot(np.linspace(example.L, example.R, example.sxNum+1), final[-3])
    plt.show()
    plt.plot(np.linspace(example.L, example.R, example.sxNum+1), final[-2])
    plt.show()
    plt.plot(np.linspace(example.L, example.R, example.sxNum+1), final[-1])
    plt.show()

if __name__=='__main__':
    FracAdvecDiffInverseENKFTest()