class ggEnKF:
    upd_a: str
    init_var: float
    N: int
    update_var = True
    fnoise_treatm: str = 'Stoch'

    def H_var(self, H, var: list[float, list, list]):
        sum = var[0] * H
        for i in range(len(var[1])):
            sum += (H @ var[1][i]) @ var[2][i].T

        return sum

    def clean_up(self, E, vars):
        if len(vars[0][1]) < 30:
            return E, vars

        time0 = time.time()

        N = E.shape[0]
        mu = np.mean(E, 0)  # Ens mean
        deviation = (E - mu).T   # Ens anomalies

        k = 0
        A = [deviation]
        B = [deviation]

        for i in range(len(vars)):
            k += vars[i][0]
            A += vars[i][1]
            B += vars[i][2]

        time1 = time.time()
        # print("time on construct A and B: ", time1 - time0)

        eigens = eigen_pow(k, A, B, N)

        time2 = time.time()
        # print("time on eigen_pow: ", time2 - time1)

        min_eigen = eigens[-1][1]
        # print("eigen value ratio: ", eigens[0][1] / min_eigen)
        # if eigens[-1][1] > eigens[0][1] / 10:
        #     raise RuntimeWarning("{}th eigenvalue is not small enough. Check your algorithm.".format(N+1))

        next_E = np.zeros_like(E)
        for i in range(N-1):
            next_E[i] = sqrt((eigens[i][1] - min_eigen) / 2) * eigens[i][2].T

        next_E[-1] = -np.sum(next_E[:-1], axis=0)
        next_E += mu

        for i in range(len(vars)):
            vars[i][0] = min_eigen / N
            vars[i][1].clear()
            vars[i][2].clear()

        time3 = time.time()
        # print("time on construct next_E: ", time3 - time2)

        return next_E, vars

    def assimilate(self, HMM, xx, yy):
        # init
        E = HMM.X0.sample(self.N)
        self.stats.assess(0, E=E)

        vars = []
        for i in range(self.N):
            var = []
            var.append(self.init_var)
            var.append(list())
            var.append(list())
            vars.append(var)

        # cycle
        for k, ko, t, dt in progbar(HMM.tseq.ticker):
            # print(k)
            # propagate
            E = HMM.Dyn(E, t-dt, dt)
            if self.update_var:
                E = add_noise(E, dt, HMM.Dyn.noise, self.fnoise_treatm)
                for i in range(self.N):
                    # print(HMM.Dyn)
                    D = eye(E.shape[1]) + dt * HMM.Dyn.linear(E[i], t, dt)
                    for j in range(len(vars[i][1])):
                        vars[i][1][j] = D @ vars[i][1][j]
                        vars[i][2][j] = D @ vars[i][2][j]

                    if abs(vars[i][0]) > 1e-3:
                        vars[i][1].append(vars[i][0] * D)
                        vars[i][2].append(D)
                        vars[i][0] = 0

            # analysis update
            if ko is not None:
                self.stats.assess(k, ko, 'f', E=E)

                # time0 = time.time()

                # first calculate the variances
                H_var = []
                H_var_H_t = []
                for i in range(self.N):
                    # print("obs is :", HMM.Obs(ko))
                    H_i = HMM.Obs.Op1.linear(E[i])
                    print(H_i)
                    H_var_i = self.H_var(H_i, vars[i])
                    H_var_i_H_t = H_var_i @ H_i.T

                    H_var.append(H_var_i)
                    H_var_H_t.append(H_var_i_H_t)

                sum_H_var = sum(H_var)
                sum_H_var_H_t = sum(H_var_H_t)

                time1 = time.time()
                # print("time on H_var: ", time1 - time0)

                # calculate the Kalman gain
                Eo = HMM.Obs(ko)(E)  # Ens of obs, actually (HX).t
                hnoise = HMM.Obs(ko).noise # noise of ob
                y = yy[ko]  # Anomaly

                R = hnoise.C  # Obs noise cov
                N, Nx = E.shape  # Dimensionality

                mu = np.mean(E, 0)  # Ens mean
                A = E - mu  # Ens anomalies

                xo = np.mean(Eo, 0)  # Obs ens mean
                Y = Eo - xo  # Obs ens anomalies

                C = Y.T @ Y + sum_H_var_H_t + R.full * N
                YC = mrdiv(Y, C)
                KG = A.T @ YC + sla.solve(C, sum_H_var).T
                E += (KG @ (y - Eo).T).T

                # time2 = time.time()
                # print("time on KG: ", time2 - time1)

                # update variance
                for i in range(self.N):
                    H_var_i_t = H_var[i].T
                    vars[i][1].append(-KG)
                    vars[i][2].append(H_var_i_t)
                    vars[i][1].append(-H_var_i_t)
                    vars[i][2].append(KG)
                    vars[i][1].append(KG @ (H_var_H_t[i] + R.full))
                    vars[i][2].append(KG)

                # time3 = time.time()
                # print("time on update var: ", time3 - time2)

                E, vars = self.clean_up(E, vars)

                # time4 = time.time()
                # print("time on clean up: ", time4 - time3)

            self.stats.assess(k, ko, E=E)
