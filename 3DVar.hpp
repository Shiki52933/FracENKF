#pragma once
#include "utility.hpp"

namespace shiki
{

    class Var3d
    {
        bool is_fractional;
        int len_window;
        arma::mat var;

    public:
        std::vector<arma::vec> res;
        std::vector<double> max_error, relative_error;

    public:
        Var3d(bool is_fractional, arma::mat var, int len_window = 1)
            : is_fractional(is_fractional), var(var), len_window(len_window)
        {
        }

        void assimilate(
            int iters_num, double t, double dt,
            arma::vec mean, HMM &hmm, Observer &ob)
        {
            res.clear();
            max_error.clear();
            relative_error.clear();

            for (int i = 0; i < iters_num; i++)
            {
                if (ob.is_observable(t))
                {
                    int dim = mean.n_rows;
                    arma::mat core = var.submat(0, 0, dim - 1, dim - 1);
                    arma::vec y = ob.get_observation(t);
                    arma::sp_mat ob_op = ob.linear(t, mean);
                    arma::vec innovation = y - ob_op * mean;
                    if (!is_fractional)
                    {
                        arma::mat gain = core * ob_op.t() * arma::inv(ob_op * core * ob_op.t() + ob.noise(t));
                        mean += gain * innovation;
                    }
                    else
                    {
                        int pre_len = res.size() >= len_window ? len_window - 1 : res.size();
                        arma::mat fvar = var.submat(0, 0, (pre_len + 1) * dim - 1, ob_op.n_cols - 1);
                        arma::mat gain = fvar * ob_op.t() * arma::inv(ob_op * core * ob_op.t() + ob.noise(t));
                        arma::vec update = gain * innovation;

                        // 更新
                        mean += update.subvec(0, dim - 1);
                        for (int j = res.size() - pre_len; j < res.size(); j++)
                        {
                            res.at(j) += update.subvec(dim * (res.size() - j), dim * (res.size() - j + 1) - 1);
                        }
                    }
                }

                {
                    // 储存结果
                    res.push_back(mean);
                    arma::vec error = arma::abs(hmm.get_state(t) - mean);
                    max_error.push_back(arma::max(error));
                    relative_error.push_back(arma::norm(error) / (arma::norm(hmm.get_state(t)) + 1e-6));
                }

                // 如果不是最后一步，就往前推进
                if (i != iters_num - 1)
                {
                    if (is_fractional)
                    {
                        arma::vec history = res.back();
                        for (int i = res.size() - 2; i >= 0; i--)
                        {
                            history = arma::join_cols(history, res[i]);
                        }
                        mean = hmm.model(t, dt, history);
                    }
                    else
                    {
                        mean = hmm.model(t, dt, mean);
                    }
                    t += dt;
                }
            }
        }
    };
}