#pragma once
#include "utility.hpp"

namespace shiki
{

    class fEnKF
    {
    public:
        int len_window;
        std::vector<double> max_error, relative_error;
        std::vector<arma::vec> res;
        std::vector<arma::mat> history;

    public:
        fEnKF(int len_window) : len_window(len_window)
        {
        }

        arma::mat get_full_ensemble(const arma::mat &ensemble){
            int pre_len = history.size() >= len_window ? len_window-1 : history.size();
            arma::mat full_ensemble((pre_len + 1) * ensemble.n_rows, ensemble.n_cols, arma::fill::none);
            full_ensemble.submat(0, 0, ensemble.n_rows - 1, ensemble.n_cols - 1) = ensemble;
            for (int j = 0; j < pre_len; j++)
            {
                full_ensemble.submat((j + 1) * ensemble.n_rows, 0, (j + 2) * ensemble.n_rows - 1, ensemble.n_cols - 1) = history[history.size() - j - 1];
            }
            return full_ensemble;
        }

        arma::mat total_history(){
            int dim = history[0].n_rows;
            arma::mat total(dim * history.size(), history[0].n_cols, arma::fill::none);
            for(int i=0;i<history.size();i++){
                // 时间逆序
                total.submat((history.size() - i - 1) * dim, 0, (history.size() - i) * dim - 1, history[0].n_cols - 1) = history[i];
            }
            return total;
        }

        void update_history(const arma::mat &ensemble, const arma::mat &full_ensemble){
            int dim = ensemble.n_rows;
            int pre_len = full_ensemble.n_rows / dim - 1;
            for(int j=0;j<pre_len;j++)
            {
                history[history.size() - j - 1] = full_ensemble.submat((j + 1) * dim, 0, (j + 2) * dim - 1, ensemble.n_cols - 1);
            }
        }

        void assimilate(
            int iters_num, double t0, double dt,
            arma::mat ensemble, BHMM &hmm, Observer &ob)
        {
            // prepare
            max_error.clear();
            relative_error.clear();
            res.clear();
            history.clear();
            int ensemble_size = ensemble.n_cols;

            for (int i = 0; i < iters_num; i++)
            {
                if (ob.is_observable(t0))
                {
                    arma::vec y = ob.get_observation(t0);
                    int ob_size = y.n_rows;

                    // 生成扰动后的观测
                    arma::mat perturb;
                    arma::mvnrnd(perturb, arma::vec(ob_size), ob.noise(t0), ensemble_size);
                    arma::mat after_perturb = perturb.each_col() + y;

                    // 观测后集合
                    arma::mat y_f = ob.observe(t0, ensemble);
                    arma::mat auxiliary = after_perturb - y_f;

                    // 平均值
                    arma::mat ensemble_mean = arma::mean(ensemble, 1);
                    arma::mat core = (ensemble.each_col() - ensemble_mean) / sqrt(ensemble_size - 1);

                    // 组装历史
                    arma::mat full_ensemble = get_full_ensemble(ensemble);
                    arma::vec full_mean = arma::mean(full_ensemble, 1);
                    arma::mat full_core = (full_ensemble.each_col() - full_mean) / sqrt(ensemble_size - 1);

                    // 计算增益
                    arma::mat H(ob.linear(t0, ensemble_mean));
                    arma::mat H_core = H * core;
                    arma::mat gain = full_core * H_core.t() * arma::inv(H_core * H_core.t() + ob.noise(t0));

                    // 更新集合
                    full_ensemble += gain * auxiliary;

                    // 更新历史
                    ensemble = full_ensemble.submat(0, 0, ensemble.n_rows - 1, ensemble_size - 1);
                    update_history(ensemble, full_ensemble);
                }

                {
                    history.push_back(ensemble);
                    // 储存结果
                    arma::vec mu = arma::mean(ensemble, 1);
                    res.push_back(mu);
                    arma::vec error = arma::abs(hmm.get_state(t0) - mu);
                    max_error.push_back(arma::max(error));
                    relative_error.push_back(arma::norm(error) / (arma::norm(hmm.get_state(t0)) + 1e-6));
                }

                // 如果不是最后一步，就往前推进
                if (i != iters_num - 1)
                {
                    arma::mat total = total_history();
                    ensemble = hmm.model(t0, dt, total);
                    t0 += dt;
                }
            }
        }
    };
}