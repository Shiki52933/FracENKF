#pragma once
#include "utility.hpp"
#include "eigen.hpp"

namespace shiki
{
    /// @brief gEnKF, original version, for small scale problems
    class gEnKF
    {
    public:
        std::vector<arma::vec> res;
        std::vector<double> max_error, relative_error;

        void assimilate(
            int iters_num, double t0, double dt,
            arma::mat ensemble, std::vector<arma::mat> vars,
            Observer &ob, BHMM &hmm)
        {
            // 初始化
            max_error.clear();
            relative_error.clear();
            res.clear();
            int ensemble_size = ensemble.n_cols;
            arma::mat I = arma::eye(ensemble.n_rows, ensemble.n_rows);

            for (int i = 0; i < iters_num; i++)
            {
                if (ob.is_observable(t0))
                {
                    std::vector<arma::sp_mat> Hs;
                    for (int j = 0; j < ensemble_size; ++j)
                    {
                        Hs.push_back(ob.linear(t0, ensemble.col(j)));
                    }

                    // Var_i
                    arma::mat HVar = Hs[0] * vars[0];
                    arma::mat HVarH = HVar * Hs[0].t();
                    for (int j = 1; j < vars.size(); ++j)
                    {
                        HVar += Hs[j] * vars[j];
                        HVarH += Hs[j] * vars[j] * Hs[j].t();
                    }
                    HVar /= vars.size();
                    HVarH /= vars.size();

                    // 平均值
                    arma::mat ensemble_mean = arma::mean(ensemble, 1);
                    arma::mat x_f = (ensemble.each_col() - ensemble_mean) / sqrt(ensemble_size - 1);
                    arma::mat Yf = ob.observe(t0, ensemble);
                    arma::mat Yf_mean = arma::mean(Yf, 1);
                    arma::mat y_f = (Yf.each_col() - Yf_mean) / sqrt(ensemble_size - 1);
                    arma::mat P_f = y_f * y_f.t() + HVarH;

                    // 计算增益矩阵
                    arma::mat gain = (x_f * y_f.t() + HVar.t()) * arma::inv(P_f + ob.noise(t0));

                    // 分析步
                    ensemble += gain * (ob.get_observation(t0) - Yf.each_col());

                    // 更新vars
                    arma::mat change = gain * ob.noise(t0) * gain.t();
                    for (int j = 0; j < vars.size(); ++j)
                    {
                        arma::mat coef = I - gain * Hs[j];
                        vars[j] = coef * vars[j] * coef.t() + change;
                    }
                }

                {
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
                    for (int j = 0; j < ensemble_size; ++j)
                    {
                        arma::sp_mat D = hmm.linear(t0, ensemble.col(j));
                        // 是否将sys_error放在这有待商榷
                        // 个人认为不要放在这
                        // vars[j] += h * h * (D * vars[j] * D.t() + sys_errors[i]);
                        // vars[j] += h * h * (D * vars[j] * D.t());
                        vars[j] = (I + dt * D) * vars[j] * (I + dt * D).t() + hmm.noise(t0);
                        // vars[j] += dt * dt * I;
                    }
                    ensemble = hmm.model(t0, dt, ensemble);
                    // ensemble = model(ensemble, i, sys_errors[i]);

                    t0 += dt;
                }
            }

            return;
        }

        /// @brief  H is linear and time-variant
        void assimilate_linear(
            int iters_num, double t0, double dt,
            arma::mat ensemble, std::vector<arma::mat> vars,
            Observer &ob, BHMM &hmm)
        {
            // 初始化
            max_error.clear();
            relative_error.clear();
            res.clear();
            int ensemble_size = ensemble.n_cols;
            arma::mat I = arma::eye(ensemble.n_rows, ensemble.n_rows);

            for (int i = 0; i < iters_num; i++)
            {
                if (ob.is_observable(t0))
                {
                    arma::sp_mat H = ob.linear(t0, ensemble.col(0));

                    // Var_i
                    arma::mat Var = vars[0];
                    for (int j = 1; j < vars.size(); ++j)
                    {
                        Var += vars[j];
                    }
                    Var /= vars.size();

                    // 平均值
                    arma::mat ensemble_mean = arma::mean(ensemble, 1);
                    arma::mat x_f = (ensemble.each_col() - ensemble_mean) / sqrt(ensemble_size - 1);
                    arma::mat P_f = x_f * x_f.t() + Var;

                    // 计算增益矩阵
                    arma::mat gain = P_f * H.t() * arma::inv(H * P_f * H.t() + ob.noise(t0));

                    // 分析步
                    ensemble += gain * (ob.get_observation(t0) - (H * ensemble).each_col());

                    // 更新vars
                    arma::mat coef = I - gain * H;
                    arma::mat change = gain * ob.noise(t0) * gain.t();
                    for (int j = 0; j < vars.size(); ++j)
                    {
                        vars[j] = coef * vars[j] * coef.t() + change;
                    }
                }

                {
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
                    for (int j = 0; j < ensemble_size; ++j)
                    {
                        arma::sp_mat D = hmm.linear(t0, ensemble.col(j));
                        // 是否将sys_error放在这有待商榷
                        // 个人认为不要放在这
                        // vars[j] += h * h * (D * vars[j] * D.t() + sys_errors[i]);
                        // vars[j] += h * h * (D * vars[j] * D.t());
                        vars[j] = (I + dt * D) * vars[j] * (I + dt * D).t() + hmm.noise(t0);
                        // vars[j] += dt * dt * I;
                    }
                    ensemble = hmm.model(t0, dt, ensemble);
                    // ensemble = model(ensemble, i, sys_errors[i]);

                    t0 += dt;
                }
            }

            return;
        }
    };

    /// @brief gEnKF for problems where H is linear and time-invariant
    class gEnKF_H
    {
    public:
        std::vector<double> max_error, relative_error;

        void assimilate(
            int iters_num, double t0, double dt,
            arma::mat ensemble, std::vector<arma::sp_mat> Hvars,
            Observer &ob, BHMM &hmm)
        {
            // prepare
            max_error.clear();
            relative_error.clear();
            int en_size = ensemble.n_cols;
            arma::sp_mat H = ob.linear(t0, ensemble.col(0));

            for (int i = 0; i < iters_num; ++i)
            {
                arma::vec X_mean = arma::mean(ensemble, 1);
                if (ob.is_observable(t0))
                {
                    arma::mat Xf = (ensemble.each_col() - X_mean) / sqrt(en_size);
                    // arma::mat HX = H * ensemble; // size = ob_size * ensemble_size, affordable
                    // arma::mat HXf = H * Xf; // size = ob_size * ensemble_size, affordable
                    arma::mat HXf = H * Xf; // size = ob_size * ensemble_size, affordable

                    arma::sp_mat mean_var = Hvars[0];
                    for (int j = 1; j < Hvars.size(); ++j)
                    {
                        mean_var += Hvars[j];
                    }
                    mean_var /= Hvars.size();

                    arma::mat rmat = arma::inv_sympd(HXf * HXf.t() + mean_var * H.t() + ob.noise(t0)); // size = ob_size * ob_size, small
                    arma::mat gain = Xf * (HXf.t() * rmat) + mean_var.t() * rmat;                      // size = pde_size * ob_size, affordable

                    // update states
                    ensemble += gain * (ob.get_observation(t0) - (H * ensemble).each_col());

                    // update variances
                    arma::mat H_gain = H * gain; // size = ob_size * ob_size, small
                    arma::mat add_on = H_gain * ob.noise(t0) * gain.t();
                    for (auto &Hvar : Hvars)
                    {
                        arma::mat H_var_H_t(Hvar * H.t()); // size = ob_size * ob_size, small
                        Hvar = Hvar - H_gain * Hvar - H_var_H_t * gain.t() + (H_gain * H_var_H_t) * gain.t() + add_on;
                    }

                    // update mean
                    X_mean = arma::mean(ensemble, 1);
                }

                {
                    // check error and save results
                    // read from file == filename
                    arma::vec reference = hmm.get_state(t0);
                    arma::vec error = arma::abs(reference - X_mean);
                    max_error.push_back(arma::max(error));
                    relative_error.push_back(arma::norm(error) / (arma::norm(reference) + 1e-6));
                }

                if (i != iters_num - 1)
                {
                    ensemble = hmm.model(t0, dt, ensemble);
                    // update step, we dont change vars here for simplicity
                    // vars[j] *= (1 + dt);
                    t0 += dt;
                }
            }

            return;
        };
    };

    /// @brief gEnKF for large scale problems in pde
    class ggEnKF
    {
        double init_var;
        int N;
        bool update_var;

    public:
        std::vector<double> max_error, relative_error, eigen_ratios;

    public:
        ggEnKF(double init_var, int N, bool update_var = true) : init_var(init_var), N(N), update_var(update_var) {}

    private:
        arma::mat static H_var(const arma::sp_mat &H, const GVar &var)
        {
            arma::mat total(var.k * H);
            for (int i = 0; i < var.A.size(); ++i)
            {
                total += (H * var.A[i]) * var.B[i].t();
            }
            return total;
        }

        void trim_vars(std::vector<GVar> &vars)
        {
            int cols_save_max = 400;
            double min_eigen_ratio = 1e6;
            for (auto &var : vars)
            {
                int cols_sum = 0;
                for (int i = 0; i < var.A.size(); ++i)
                {
                    cols_sum += var.A[i].n_cols;
                }
                if (cols_sum < 2 * cols_save_max)
                    continue;

                auto [eigen_value, eigen_vector] = eigen_pow_opt(var.k, var.A, var.B);
                // semi positive can be garanteed, so sqrt is safe
                eigen_vector *= arma::diagmat(arma::sqrt(eigen_value));
                int cols_save = std::min(cols_save_max, (int)eigen_value.n_elem);
                eigen_vector = eigen_vector.cols(0, cols_save - 1);

                double eigen_ratio = eigen_value(0) / (eigen_value(cols_save - 1) + 1e-6);
                min_eigen_ratio = std::min(min_eigen_ratio, eigen_ratio);

                var.A.clear();
                var.B.clear();
                var.A.push_back(eigen_vector);
                var.B.push_back(eigen_vector);
                var.k = eigen_value(cols_save - 1);
            }
            std::cout << "min eigen ratio: " << min_eigen_ratio << std::endl;
            eigen_ratios.push_back(min_eigen_ratio);
        }

        void trim_vars_no_k(std::vector<GVar> &vars)
        {
            int cols_save_max = 400;
            double min_eigen_ratio = 1e6;
            for (auto &var : vars)
            {
                int cols_sum = 0;
                for (int i = 0; i < var.A.size(); ++i)
                {
                    cols_sum += var.A[i].n_cols;
                }
                if (cols_sum < 2 * cols_save_max)
                    continue;

                auto [eigen_value, eigen_vector] = eigen_pow_opt(0, var.A, var.B);
                // semi positive can be garanteed, so sqrt is safe
                eigen_vector *= arma::diagmat(arma::sqrt(eigen_value));
                int cols_save = std::min(cols_save_max, (int)eigen_value.n_elem);
                eigen_vector = eigen_vector.cols(0, cols_save - 1);

                double eigen_ratio = arma::sum(eigen_value.subvec(0, cols_save - 1)) / (arma::sum(eigen_value) + 1e-6);
                min_eigen_ratio = std::min(min_eigen_ratio, eigen_ratio);

                var.A.clear();
                var.B.clear();
                var.A.push_back(eigen_vector);
                var.B.push_back(eigen_vector);
            }
            std::cout << "min eigen ratio: " << min_eigen_ratio << std::endl;
            eigen_ratios.push_back(min_eigen_ratio);
        }

        void clean_up(arma::mat &E, std::vector<GVar> &vars)
        {
            if (vars[0].A.size() < 10)
                return;

            int N = E.n_cols;
            arma::mat mu = arma::mean(E, 1);
            arma::mat deviation = E.each_col() - mu;

            GVar temp;
            temp.k = 0;
            temp.A.push_back(deviation);
            temp.B.push_back(deviation);

            if (vars.size() > 1)
            {
                for (auto &var : vars)
                {
                    temp.k += var.k;
                    for (int i = 0; i < var.A.size(); ++i)
                    {
                        temp.A.push_back(var.A[i]);
                        temp.B.push_back(var.B[i]);
                    }
                }
            }
            else if (vars.size() == 1)
            {
                temp.k = vars[0].k * N;
                for (int i = 0; i < vars[0].A.size(); ++i)
                {
                    temp.A.push_back(vars[0].A[i] * sqrt(N));
                    temp.B.push_back(vars[0].B[i] * sqrt(N));
                }
            }

            auto [eigen_value, eigen_vector] = eigen_pow_opt(temp.k, temp.A, temp.B, N);
            double min_eigen = arma::min(eigen_value);
            double max_eigen = arma::max(eigen_value);
            double ratio = max_eigen / (min_eigen + 1e-6);
            // std::cout << "eigen ratio: " << ratio << std::endl;
            eigen_ratios.push_back(ratio);

            for (int i = 0; i < N - 1; ++i)
            {
                // E.col(i) = sqrt((eigen_value(i) - min_eigen)/2) * eigen_vector.col(i) + mu;
                E.col(i) = sqrt(eigen_value(i)) * eigen_vector.col(i) + mu;
            }
            E.col(N - 1) = N * mu - arma::sum(E.submat(0, 0, E.n_rows - 1, N - 2), 1);

            assert(arma::norm(arma::mean(E, 1) - mu) < 1e-6);

            for (auto &var : vars)
            {
                var.k = min_eigen / N;
                var.A.clear();
                var.B.clear();
            }
        }

    public:
        void assimilate(
            int iters_num, double t0, double dt,
            arma::mat ensemble, std::vector<GVar> vars,
            Observer &ob, BHMM &hmm)
        {
            using std::vector;
            const int N = ensemble.n_cols;
            max_error.clear();
            relative_error.clear();
            eigen_ratios.clear();

            for (int i = 0; i < iters_num; i++)
            {
                if (ob.is_observable(t0))
                {
                    // 计算局部方差
                    arma::mat sum_H_var, sum_H_var_H_t;
                    vector<arma::mat> H_var_list;
                    vector<arma::mat> H_var_H_t_list;
                    for (int j = 0; j < vars.size(); ++j)
                    {
                        arma::sp_mat H_j = ob.linear(t0, ensemble.col(j));
                        arma::mat H_var_j = H_var(H_j, vars[j]);
                        arma::mat H_var_j_H_t = H_var_j * H_j.t();
                        H_var_list.push_back(H_var_j);
                        H_var_H_t_list.push_back(H_var_j_H_t);
                        if (j == 0)
                        {
                            sum_H_var = H_var_j;
                            sum_H_var_H_t = H_var_j_H_t;
                        }
                        else
                        {
                            sum_H_var += H_var_j;
                            sum_H_var_H_t += H_var_j_H_t;
                        }
                    }
                    // back door
                    if (vars.size() == 1)
                    {
                        // ob is independant of position
                        sum_H_var *= N;
                        sum_H_var_H_t *= N;
                    }

                    // -------------------------------------
                    auto Eo = ob.observe(t0, ensemble);     // HX
                    arma::mat Eo_mu = arma::mean(Eo, 1);    // HX_mean
                    arma::mat Hx_f = Eo.each_col() - Eo_mu; // HX_f

                    // 平均值
                    arma::mat mu = arma::mean(ensemble, 1);
                    arma::mat x_f = ensemble.each_col() - mu;

                    arma::mat ob_noise = ob.noise(t0);
                    arma::mat C = arma::inv_sympd(Hx_f * Hx_f.t() + sum_H_var_H_t + ob_noise * N);
                    arma::mat KG = x_f * (Hx_f.t() * C) + sum_H_var.t() * C;

                    auto y = ob.get_observation(t0);
                    ensemble += KG * (y - Eo.each_col());

                    // 更新vars
                    for (int j = 0; j < vars.size(); ++j)
                    {
                        arma::mat H_var_j_t = H_var_list[j].t();
                        arma::mat t = H_var_H_t_list[j] + ob_noise;
                        t = KG * t;
                        vars[j].A.push_back(KG);
                        vars[j].B.push_back(-H_var_j_t + t);
                        vars[j].A.push_back(-H_var_j_t);
                        vars[j].B.push_back(KG);
                    }

                    // clean_up(ensemble, vars);
                    trim_vars_no_k(vars);
                }
                
                {
                    // 储存结果
                    arma::vec mu = arma::mean(ensemble, 1);
                    // res.push_back(vec(mean(ensemble, 1)));
                    // check error
                    arma::vec real = hmm.get_state(t0);
                    max_error.push_back(arma::max(arma::abs(mu - real)));
                    relative_error.push_back(arma::norm(mu - real) / (arma::norm(real) + 1e-6));
                }

                // 如果不是最后一步，就往前推进
                if (i != iters_num - 1)
                {
                    ensemble = hmm.model(t0, dt, ensemble);
                    t0 += dt;

                    // update variances
                    if (this->update_var)
                    {
                        for (auto &var : vars)
                        {
                            var.k = var.k * (1 + dt) * (1 + dt);
                            for (int i = 0; i < var.A.size(); ++i)
                            {
                                var.A[i] *= (1 + dt);
                                var.B[i] *= (1 + dt);
                            }
                        }
                    }
                }
            }

            // return errors;
        }
    };

}