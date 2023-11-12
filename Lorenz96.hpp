#pragma once
#include "utility.hpp"

namespace shiki
{
    class Lorenz96Model : public HMM
    {
        int dim = 40;
        double F = 8;
        double dt = 0.005;
        double t_max = 20;
        arma::mat sys_var;
        arma::vec s;
        std::vector<arma::vec> states;
        std::vector<double> times;

    public:
        Lorenz96Model(int dim, double F, double dt, double t_max, arma::mat sys_var) : dim(dim), F(F), dt(dt), t_max(t_max), sys_var(sys_var)
        {
            assert(sys_var.n_rows == dim && sys_var.n_cols == dim);
            s = arma::randn(dim);
        }

        void set_state(arma::vec s)
        {
            this->s = s;
        }

        arma::mat model(double t, double dt, const arma::mat &ensemble) override
        {
            int M = ensemble.n_rows, N = ensemble.n_cols;
            arma::mat newer(M, N, arma::fill::none);

            for (int i = 0; i < M; i++)
            {
                int pre = (i + M - 1) % M;
                int far = (i + M - 2) % M;
                int next = (i + 1) % M;

                arma::mat derivative = ensemble.row(pre) % (ensemble.row(next) - ensemble.row(far)) - ensemble.row(i) + F;
                arma::mat value = derivative * dt + ensemble.row(i);
                newer.row(i) = value;
            }

            // add perturbation
            arma::mat perturb;
            if (arma::mvnrnd(perturb, arma::vec(M), sys_var, N))
            {
                newer += dt * perturb;
            }

            return newer;
        }

        arma::sp_mat linear(double t, const arma::vec &mean) override
        {
            int dim = mean.n_rows;
            arma::mat derivative(dim, dim, arma::fill::zeros);
            for (int i = 0; i < dim; i++)
            {
                int pre = (i + dim - 1) % dim;
                int far = (i + dim - 2) % dim;
                int next = (i + 1) % dim;

                derivative(i, far) = -mean(pre);
                derivative(i, pre) = mean(next) - mean(far);
                derivative(i, i) = -1;
                derivative(i, next) = mean(pre);
            }

            return arma::sp_mat(derivative);
        }

        arma::sp_mat noise(double t) override
        {
            return dt * dt * arma::sp_mat(sys_var);
        }

        void reference() override
        {
            states.clear();
            times.clear();
            states.push_back(s);
            times.push_back(0);
            for (double t = 0; t < t_max; t += dt)
            {
                s = model(t, dt, s);
                states.push_back(s);
                times.push_back(t + dt);
            }
        }

        arma::vec get_state(double t) override
        {
            int index = (int)(t / dt);
            return states.at(index);
        }

        std::vector<double> get_times() override
        {
            return times;
        }
    };

}