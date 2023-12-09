#pragma once
#include "utility.hpp"

namespace shiki
{
    class fgEnKF
    {
        int len_window;
        arma::drowvec orders;
        std::vector<arma::sp_mat> frac_deri_mats;

    public:
        std::vector<arma::vec> res;
        std::vector<double> max_error, relative_error;

    public:
        fgEnKF(int len_window, arma::drowvec orders) : len_window(len_window), orders(orders)
        {
            auto binos = compute_bino(orders, len_window + 5);
            for (int i = 0; i < len_window; ++i)
            {
                arma::sp_mat pre_i(orders.n_elem, orders.n_elem);
                for (int j = 0; j < orders.n_elem; ++j)
                {
                    pre_i(j, j) = binos(i + 1, j);
                }
                frac_deri_mats.push_back(pre_i);
            }
        }

        void assimilate(int iters_num, double t0, double dt, int dim,
                        arma::mat ensemble, std::vector<arma::mat> vars,
                        BHMM &hmm, Observer &ob)
        {
            // 初始化
            max_error.clear();
            relative_error.clear();
            res.clear();
            int ensemble_size = ensemble.n_cols;
            arma::mat I = arma::eye(ensemble.n_rows, ensemble.n_rows);

            arma::sp_mat h(orders.n_elem, orders.n_elem);
            for (int i = 0; i < orders.n_elem; ++i)
            {
                h(i, i) = std::pow(dt, orders(i));
            }

            for (int i = 0; i < iters_num; i++)
            {
                if (ob.is_observable(t0))
                {
                    arma::sp_mat H = ob.linear(t0, ensemble.col(0));

                    // Var_i
                    int total_dim = vars[0].n_rows;
                    arma::mat Var = vars[0].submat(0, 0, dim - 1, total_dim - 1);
                    for (int j = 1; j < vars.size(); ++j)
                    {
                        Var += vars[j].submat(0, 0, dim - 1, total_dim - 1);
                    }
                    Var /= vars.size();

                    // 平均值
                    arma::mat ensemble_mean = arma::mean(ensemble.submat(0, 0, total_dim - 1, ensemble.n_cols - 1), 1);
                    arma::mat x_f = (ensemble.submat(0, 0, total_dim - 1, ensemble.n_cols - 1).each_col() - ensemble_mean) / sqrt(ensemble_size - 1);
                    arma::mat H_var = H * x_f.submat(0, 0, dim - 1, x_f.n_cols - 1) * x_f.t() + H * Var;

                    // 计算增益矩阵
                    arma::mat gain = H_var.t() * arma::inv(H_var * H_var.t() + ob.noise(t0));

                    // 分析步
                    ensemble.submat(0, 0, total_dim-1, ensemble.n_cols-1) += gain * (ob.get_observation(t0) - (H * ensemble.submat(0, 0, dim - 1, ensemble.n_cols - 1)).each_col());

                    // 更新vars
                    arma::mat coef = arma::eye(total_dim, total_dim);
                    coef.submat(0, 0, total_dim - 1, dim - 1) -= gain * H;
                    arma::mat change = gain * ob.noise(t0) * gain.t();
                    for (int j = 0; j < vars.size(); ++j)
                    {
                        vars[j] = coef * vars[j] * coef.t() + change;
                    }
                }

                {
                    // 储存结果
                    arma::vec mu = arma::mean(ensemble.submat(0, 0, dim - 1, ensemble.n_cols - 1), 1);
                    res.push_back(mu);
                    arma::vec error = arma::abs(hmm.get_state(t0) - mu);
                    max_error.push_back(arma::max(error));
                    relative_error.push_back(arma::norm(error) / (arma::norm(hmm.get_state(t0)) + 1e-6));
                }

                // 如果不是最后一步，就往前推进
                if (i != iters_num - 1)
                {
                    int current_window = vars[0].n_rows / dim;
                    for (int j = 0; j < ensemble_size; ++j)
                    {
                        arma::sp_mat D = hmm.linear(t0, ensemble.col(j).subvec(0, dim - 1));
                        arma::sp_mat propa(dim, vars[j].n_cols);
                        propa.submat(0, 0, dim - 1, dim - 1) = h * D - frac_deri_mats[0];
                        for (int k = 1; k < current_window; ++k)
                        {
                            propa.submat(0, k * dim, dim - 1, (k + 1) * dim - 1) = -frac_deri_mats[k];
                        }
                        arma::sp_mat lower = arma::eye<arma::sp_mat>(vars[j].n_rows, vars[j].n_rows);
                        arma::sp_mat total_D = arma::join_cols(propa, lower);
                        vars[j] = total_D * vars[j] * total_D.t();
                        // vars[j].submat(0, 0, dim - 1, dim - 1) += h * h * hmm.noise(t0);
                        if (current_window == len_window)
                        {
                            vars[j] = vars[j].submat(0, 0, len_window * dim - 1, len_window * dim - 1);
                        }
                    }
                    arma::mat next = hmm.model(t0, dt, ensemble);
                    ensemble = arma::join_cols(next, ensemble);
                    // ensemble = model(ensemble, i, sys_errors[i]);

                    t0 += dt;
                }
            }
        };
    };
}