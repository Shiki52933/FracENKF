#pragma once
#include "utility.hpp"

namespace shiki
{
    class FLornez96Model : public HMM
    {
        int dim = 40;
        double F = 8;
        double dt = 0.005;
        double t_max = 20;
        arma::mat sys_var;
        

        std::vector<arma::vec> states;
        std::vector<double> times;

    public:
        arma::vec s;
        arma::drowvec orders;
        arma::mat binos;

    public:
        FLornez96Model(int dim, double F, double dt, double t_max, arma::mat sys_var) : dim(dim), F(F), dt(dt), t_max(t_max), sys_var(sys_var)
        {
            assert(sys_var.n_rows == dim && sys_var.n_cols == dim);
            s = arma::randn(dim);
            orders = 1. + 0.03 * arma::randn(dim).t();
            binos = compute_bino(orders, t_max / dt + 5);
        }

        void set_state(arma::vec s)
        {
            this->s = s;
        }

        arma::mat model(double t, double dt, const arma::mat &ensemble) override
        {
            // 从上到下时间从近到远
            int window = ensemble.n_rows / dim;

            // 首先计算分数阶导数
            arma::mat frac_dirivative(dim, ensemble.n_cols, arma::fill::zeros);
            for (int i = 0; i < window; i++)
            {
                for (int idx = 0; idx < dim; idx++)
                {
                    frac_dirivative.row(idx) += ensemble.row(i * dim + idx) * binos(i + 1, idx);
                }
            }

            // 计算右端项
            arma::mat rhs(dim, ensemble.n_cols, arma::fill::none);
            for (int i = 0; i < dim; i++)
            {
                int pre = (i + dim - 1) % dim;
                int far = (i + dim - 2) % dim;
                int next = (i + 1) % dim;

                rhs.row(i) = ensemble.row(pre) % (ensemble.row(next) - ensemble.row(far)) - ensemble.row(i) + F;
            }

            // add perturbation
            arma::mat perturb;
            if (arma::mvnrnd(perturb, arma::vec(dim), sys_var, rhs.n_cols))
            {
                rhs += perturb;
            }

            for (int i = 0; i < dim; i++)
            {
                rhs.row(i) *= pow(dt, orders[i]);
            }

            return rhs - frac_dirivative;
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
            return arma::sp_mat(sys_var);
        }

        void reference() override
        {
            states.clear();
            times.clear();
            states.push_back(s);
            times.push_back(0);
            for (double t = 0; t < t_max; t += dt)
            {
                arma::vec next = model(t, dt, s);
                states.push_back(next);
                times.push_back(t + dt);
                s = arma::join_cols(next, s);
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

        arma::mat rhs(const arma::mat &ensemble)
        {
            // 计算右端项
            arma::mat rhs(dim, ensemble.n_cols, arma::fill::none);
            for (int i = 0; i < dim; i++)
            {
                int pre = (i + dim - 1) % dim;
                int far = (i + dim - 2) % dim;
                int next = (i + 1) % dim;

                rhs.row(i) = ensemble.row(pre) % (ensemble.row(next) - ensemble.row(far)) - ensemble.row(i) + F;
            }
            return rhs;
        }
    };
}