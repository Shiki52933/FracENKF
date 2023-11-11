#pragma once
#include "utility.hpp"
#include "StochasticENKF.hpp"


namespace shiki
{
    class Lorenz96Model: public BHMM
    {
        int dim = 40;
        double F = 8;
        double dt = 0.005;
        double t_max = 20;
        mat sys_var;
        vec s;
        std::vector<vec> states;
        std::vector<double> times;

    public:
        Lorenz96Model(int dim, double F, double dt, double t_max, mat sys_var) : dim(dim), F(F), dt(dt), t_max(t_max), sys_var(sys_var)
        {
            assert (sys_var.n_rows == dim && sys_var.n_cols == dim);
            s = arma::randn(dim);
        }

        void set_state(vec s)
        {
            this->s = s;
        }

        mat model(double t, double dt, const mat &ensemble) override
        {
            int M = ensemble.n_rows, N = ensemble.n_cols;
            mat newer(M, N, arma::fill::none);

            for (int i = 0; i < M; i++)
            {
                int pre = (i + M - 1) % M;
                int far = (i + M - 2) % M;
                int next = (i + 1) % M;

                mat derivative = ensemble.row(pre) % (ensemble.row(next) - ensemble.row(far)) - ensemble.row(i) + F;
                mat value = derivative * dt + ensemble.row(i);
                newer.row(i) = value;
            }

            // // add perturbation
            // mat temp, perturb(M, N, arma::fill::none);
            // if (arma::inv(temp, sys_var))
            // {
            //     perturb = arma::mvnrnd(vec(M, arma::fill::zeros), sys_var, N);
            //     newer += perturb;
            // }

            return newer;
        }
    
        sp_mat noise(double t) override
        {
            return sp_mat(sys_var);
        }

        void assimilate() override
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

        vec get_state(double t) override
        {
            int index = (int)(t / dt);
            return states.at(index);
        }

        std::vector<double> get_times() override
        {
            return times;
        }
    };

    class GeneralRandomObserveHelper
    {
        int dim;
        int obs_num;
        
    public:
        GeneralRandomObserveHelper(int dim, int obs_num) : dim(dim), obs_num(obs_num)
        {
        }

        sp_mat generate()
        {
            sp_mat H(obs_num, dim);
            int max_offset = dim / obs_num;
            int offset = rand() % max_offset;
            for (int i = 0; i < obs_num; i++)
            {
                H(i, offset) = 1;
                offset += max_offset;
            }
            return H;
        }
    };

}