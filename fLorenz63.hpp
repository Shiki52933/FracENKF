#pragma once
#include "utility.hpp"

namespace shiki
{
    class FLorenz63Model : public HMM
    {
        arma::drowvec orders;
        double sigma, rho, beta;
        double dt, tmax;
        arma::mat sys_var;

        std::vector<double> times;
        arma::mat binos;
        std::vector<arma::vec> states;

    public:
        FLorenz63Model(arma::drowvec orders, double sigma, double rho, double beta, double dt, double tmax, arma::mat sys_var)
            : orders(orders), sigma(sigma), rho(rho), beta(beta), dt(dt), tmax(tmax), sys_var(sys_var)
        {
            assert(orders.n_cols == 3);
            int window = tmax / dt + 5;
            binos = compute_bino(orders, window);
        }

        void set_init_state(arma::vec s)
        {
            assert(s.n_rows == 3);
            states.clear();
            states.push_back(s);
        }

        arma::mat model(double t, double dt, const arma::mat &e) override
        {
            // 从上到下时间从近到远
            int dim = 3;
            int window = e.n_rows / dim;

            // 首先计算分数阶导数
            arma::mat frac_dirivative(dim, e.n_cols, arma::fill::zeros);
            for (int i = 0; i < window; i++)
            {
                for (int idx = 0; idx < dim; idx++)
                {
                    frac_dirivative.row(idx) += e.row(i * dim + idx) * binos(i + 1, idx);
                }
            }

            // 计算右端项
            arma::mat rhs(dim, e.n_cols, arma::fill::none);
            rhs.row(0) = sigma * (e.row(1) - e.row(0));
            rhs.row(1) = e.row(0) % (rho - e.row(2)) - e.row(1);
            rhs.row(2) = e.row(0) % e.row(1) - beta * e.row(2);

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

        arma::mat rhs(const arma::mat &e)
        {
            const int dim = 3;
            arma::mat rhs(dim, e.n_cols, arma::fill::none);
            rhs.row(0) = sigma * (e.row(1) - e.row(0));
            rhs.row(1) = e.row(0) % (rho - e.row(2)) - e.row(1);
            rhs.row(2) = e.row(0) % e.row(1) - beta * e.row(2);
            return rhs;
        }

        void reference() override
        {
            times.clear();

            double t = 0;
            times.push_back(t);
            arma::vec s = states.back();
            while (t < tmax)
            {
                arma::vec next = model(t, dt, s);
                states.push_back(next);
                s = arma::join_cols(next, s);
                t += dt;
                times.push_back(t);
            }
        }

        arma::vec get_state(double t) override
        {
            int index = (int)(t / dt);
            return states.at(index);
        }

        std::vector<double> get_times()
        {
            return times;
        }

        arma::sp_mat noise(double t)
        {
            return arma::sp_mat(sys_var);
        }

        arma::sp_mat linear(double t, const arma::vec &e)
        {
            arma::mat l{
                {-sigma, sigma, 0},
                {rho - e(2), -1, -e(0)},
                {e(1), e(0), -beta}};
            return arma::sp_mat(l);
        }
    };
}