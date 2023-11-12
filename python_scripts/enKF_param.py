import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


class ObserveErrorMatrices(object):
    def __init__(self, Matrices) -> None:
        """param: Matrices is a list of numpy.ndarrays, where each is a observation error matrix, ordered by time"""
        self.idx = -1
        self.Matrices = Matrices

    def get_next(self):
        self.idx += 1
        if self.idx < len(self.Matrices):
            return self.Matrices[self.idx]
        else:
            return None


class ObserveOperator(object):
    def __init__(self, operator) -> None:
        self.operator = operator

    def apply(self, x):
        return self.operator(x)


class ModelOperator(object):
    def __init__(self, operator) -> None:
        self.operator = operator

    def apply(self, x):
        return self.operator(x)


def StochasticEnsembleKF(num_ensemble: int, init_ave: np.ndarray, init_uncertainty: np.ndarray, observations: list,
                         iters: int, Ob_Errors: ObserveErrorMatrices, Ob_Ops: ObserveOperator, Models, sys_vars):
    # initializing
    analysis_results = []
    ensemble = np.random.multivariate_normal(init_ave, init_uncertainty, size=num_ensemble).T

    for idx in range(0, iters):
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

        # test
        ensemble_mean = np.mean(ensemble_analysis, axis=1, keepdims=True)
        analysis_results.append(ensemble_mean)

        # 为了避免低效的循环，要求模型算子能够直接处理一整个集合
        ensemble = Models(ensemble_analysis, idx, sys_vars[idx])


    return analysis_results


def coefficient(len, orders):
    # 计算系数
    bino = np.ndarray([len, orders.shape[0]])
    bino[0] = 1
    for i in range(1, len):
        bino[i] = (1 - (1 + orders) / i) * bino[i - 1]
    return bino


def generate_ob_ops():
    def ob_function(input):
        H = np.array([[0.1,0.3]])
        return H.dot(input[:2])

    return ObserveOperator(ob_function)


bino = coefficient(21, np.array([0.7,1.2]))[1:]
input1 = np.array([[1.]])
input2 = np.array([[-1.]])
system_inputs = [input1 for _ in range(50)] + [input2 for _ in range(50)] \
                + [input1 for _ in range(50)] + [input2 for _ in range(50)]
system_inputs = system_inputs * 2


def model(input, count, sys_var):
    states = []
    for idx in range(input.shape[1]):
        A = np.array([[0.,1.],[-0.1, -input[-1,idx]]])
        B = np.array([[0.],[1.]])
        state = input[:-1,idx]
        state = state.reshape([state.shape[0]//2, 2])

        diff = A.dot(state[0:1].T) + B.dot(system_inputs[count])
        x_new = diff - np.sum(bino[:state.shape[0]] * state, axis=0, keepdims=True).T
        state = np.concatenate([x_new, input[:-1,idx, None]], axis=0)[:20].squeeze()
        states.append(state)

    states = np.array(states).T
    states = np.concatenate([states, input[-1:]], axis=0)
    return states


def model1(input, count, sys_var):
    states = []
    for idx in range(input.shape[1]):
        A = np.array([[0.,1.],[-0.1, -0.2]])
        B = np.array([[0.],[1.]])
        state = input[:,idx]
        state = state.reshape([state.shape[0]//2, 2])

        diff = A.dot(state[0:1].T) + B.dot(system_inputs[count]) + np.random.multivariate_normal(np.zeros([2]),
                                                                                                 sys_var).reshape([2,1])
        x_new = diff - np.sum(bino[:state.shape[0]] * state, axis=0, keepdims=True).T
        state = np.concatenate([x_new, input[:,idx, None]], axis=0)[:20].squeeze()
        states.append(state)
    return np.array(states).T


def model2(input, count, sys_var):
    states = []
    for idx in range(input.shape[1]):
        A = np.array([[0.,1.],[-0.1, -0.2]])
        B = np.array([[0.],[1.]])
        bino = coefficient(20 + 1, input[-2:, idx])[1:]
        state = input[:-2,idx]
        state = state.reshape([state.shape[0]//2, 2])

        diff = A.dot(state[0:1].T) + B.dot(system_inputs[count]) + np.random.multivariate_normal(np.zeros([2]),
                                                                                                 sys_var).reshape([2,1])
        x_new = diff - np.sum(bino[:state.shape[0]] * state, axis=0, keepdims=True).T
        state = np.concatenate([x_new, input[:-2,idx, None]], axis=0)[:20].squeeze()
        states.append(state)

    states = np.array(states).T
    states = np.concatenate([states, input[-2:]], axis=0)
    return states


def model3(input, count, sys_var):
    states = []
    for idx in range(input.shape[1]):
        A = np.array([[0.,1.],[-0.1, -0.2]])
        B = np.array([[0.],[1.]])
        state = input[:-1,idx]
        state = state.reshape([state.shape[0]//2, 2])

        diff = A.dot(state[0:1].T) + B.dot(system_inputs[count]) + np.random.multivariate_normal(np.zeros([2]),
                                                                                                 input[-1,idx]**2 * sys_var).reshape([2,1])
        x_new = diff - np.sum(bino[:state.shape[0]] * state, axis=0, keepdims=True).T
        state = np.concatenate([x_new, input[:-1,idx, None]], axis=0)[:20].squeeze()
        states.append(state)

    states = np.array(states).T
    states = np.concatenate([states, input[-1:]], axis=0)
    return states


def generate_data():
    x0 = np.array([[0.], [0.]])
    real_vals = x0.T.copy()
    input1 = np.array([[1.]])
    input2 = np.array([[-1.]])
    system_inputs = [input1 for _ in range(50)] + [input2 for _ in range(50)] \
                    + [input1 for _ in range(50)] + [input2 for _ in range(50)]
    system_inputs = system_inputs * 2
    orders = np.array([0.7, 1.2])
    A = np.array([[0., 1.], [-0.1, -0.2]])
    B = np.array([[0.], [1.]])
    C = np.array([[0.1, 0.3]])
    bino = coefficient(400+1, orders)[1:]

    # np.random.seed(42)
    ob = C.dot(x0) + np.random.normal(0,0.03)

    for i in range(1,199+1):
        diff = A.dot(x0) + B.dot(system_inputs[i-1]) + np.random.multivariate_normal([0,0], 0.3 * np.identity(2)).reshape([2,1])
        temp = real_vals[::-1]
        x0 = diff - np.sum(bino[:temp.shape[0]] * temp, axis=0, keepdims=True).T
        real_vals = np.concatenate([real_vals, x0.T], axis=0)

        temp = C.dot(x0) + np.random.normal(0, 0.03)
        ob = np.concatenate([ob,temp], axis=0)

    np.savez("./320KB/example1/real_vals_200.npz", x1=real_vals[:,0], x2=real_vals[:,1])
    np.savez("./320KB/example1/ob_200.npz", ob=ob)

    plt.plot(range(real_vals.shape[0]), real_vals)
    plt.show()

    # print(ob.shape)
    return ob, real_vals


if __name__=='__main__':
    ob, real = generate_data()
    ob_error = np.array([[0.03]])
    ob_errors = ObserveErrorMatrices([ob_error]*200)

    init_var = 100 * np.identity(3)
    init_var[2:,2:] = np.identity(1)
    sys_vars = [1 * np.identity(2)] * 200
    results = StochasticEnsembleKF(20, np.array([0., 0., 1.]), init_var, ob, ob.shape[0], ob_errors, generate_ob_ops(), model3, sys_vars)

    results = [result[[0,1,-1]].squeeze() for result in results]
    results = np.array(results)
    print(results[:, -1]**2)
    plt.plot(range(results.shape[0]), results[:, -1]**2)
    plt.show()
    plt.plot(range(results.shape[0]), results[:, :2])
    plt.show()
    plt.plot(range(results.shape[0]), results[:, :2]-real)
    plt.show()