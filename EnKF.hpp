#pragma once
#include "utility.hpp"

namespace shiki
{

    class EnKF
    {
    public:
        bool test_normal=false;
        bool is_linear=false;
        std::vector<double> skewnesses, kurtosises;
        std::vector<arma::vec> res;

    public:
        void assimilate(
            int iters_num, double t0, double dt,
            arma::mat ensemble, Observer &ob, BHMM &hmm)
        {
            // prepare
            skewnesses.clear();
            kurtosises.clear();
            res.clear();
            int ensemble_size = ensemble.n_cols;

            for (int i = 0; i < iters_num; i++)
            {
                if (test_normal)
                {
                    skewnesses.push_back(compute_skewness(ensemble));
                    kurtosises.push_back(compute_kurtosis(ensemble));
                }

                if (ob.is_observable(t0))
                {
                    arma::vec y = ob.get_observation(t0);
                    int ob_size = y.n_rows;

                    // 生成扰动后的观测
                    arma::mat perturb;
                    arma::mvnrnd(perturb, arma::vec(ob_size), ob.noise(t0), ensemble_size);
                    arma::mat after_perturb = perturb.each_col() + y;

                    // 平均值
                    arma::mat ensemble_mean = arma::mean(ensemble, 1);
                    arma::mat x_f = (ensemble.each_col() - ensemble_mean) / sqrt(ensemble_size - 1);

                    // 观测后集合
                    arma::mat y_f = ob.observe(t0, ensemble);
                    arma::mat auxiliary = after_perturb - y_f;

                    arma::mat gain;
                    if (is_linear)
                    {
                        arma::mat P_f = x_f * x_f.t();
                        arma::mat H(ob.linear(t0, ensemble_mean));
                        gain = P_f * H.t() * inv(H * P_f * H.t() + ob.noise(t0));
                    }
                    else
                    {
                        arma::mat perturb_mean = arma::mean(perturb, 1);

                        // 观测后集合
                        arma::mat y_mean = arma::mean(y_f, 1);
                        arma::mat temp = y_f - perturb;
                        y_f = (temp.each_col() - (y_mean - perturb_mean)) / sqrt(ensemble_size - 1);

                        // 计算增益矩阵
                        gain = x_f * y_f.t() * inv(y_f * y_f.t());
                    }
                    // 更新集合
                    ensemble += gain * auxiliary;
                }

                // 储存结果
                res.push_back(arma::vec(arma::mean(ensemble, 1)));

                // 如果不是最后一步，就往前推进
                if (i != iters_num - 1)
                {
                    ensemble = hmm.model(t0, dt, ensemble);
                }
            }
        }
    };
}