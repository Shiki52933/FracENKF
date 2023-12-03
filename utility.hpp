#pragma once
#include <armadillo>
#include <assert.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <functional>

namespace shiki
{

    class GVar
    {
    public:
        double k;
        std::vector<arma::mat> A, B;
    };

    class HiddenMarkovModel
    {
    public:
        virtual void reference() = 0;
        virtual arma::vec get_state(double t) = 0;
        virtual std::vector<double> get_times() = 0;

        // t == now, t + dt == next
        virtual arma::mat model(double t, double dt, const arma::mat &e) = 0;
        virtual arma::sp_mat noise(double t) = 0;
        virtual arma::sp_mat linear(double t, const arma::vec &e)
        {
            return arma::sp_mat(e.n_rows, e.n_rows);
        }
    };

    class Observer
    {
    public:
        virtual bool is_observable(double t) = 0;
        virtual void observe() = 0;
        virtual arma::vec get_observation(double t) = 0;
        virtual arma::vec observe(double t, arma::vec e) = 0;
        virtual arma::mat observe(double t, arma::mat e) = 0;
        virtual arma::sp_mat linear(double t, arma::vec e) = 0;
        virtual arma::mat noise(double t) = 0;
    };

    using HMM = HiddenMarkovModel;
    using BHMM = HiddenMarkovModel;

    /// @brief general linear observer with changing H and noise on different times
    class LinearObserver : public Observer
    {
    public:
        int gap;
        std::vector<double> times;
        std::vector<arma::sp_mat> Hs;
        std::vector<arma::mat> noises;
        std::vector<arma::vec> observations;
        BHMM &hmm;

    public:
        LinearObserver(int gap, BHMM &hmm) : gap(gap), hmm(hmm)
        {
            auto possible_times = hmm.get_times();
            for (int i = 1; i < possible_times.size(); i += gap)
            {
                times.push_back(possible_times[i]);
            }
        }

        int find_idx(double t)
        {
            for (int i = 0; i < times.size(); ++i)
            {
                if (std::abs(times[i] - t) < 1e-6)
                {
                    return i;
                }
            }
            return -1;
        }

        auto get_times()
        {
            return times;
        }

        void set_H_noise(std::vector<arma::sp_mat> &Hs, std::vector<arma::mat> &noises)
        {
            this->Hs = Hs;
            this->noises = noises;
            assert(Hs.size() == times.size());
            assert(noises.size() == times.size());
        }

        bool is_observable(double t) override
        {
            return find_idx(t) != -1;
        }

        void observe() override
        {
            for (int i = 0; i < times.size(); ++i)
            {
                arma::vec real = hmm.get_state(times[i]);
                arma::vec noise = arma::mvnrnd(arma::vec(noises[i].n_rows), noises[i]);
                // if(arma::norm(noise) != 0){
                //     std::cout<<"noise is not zero"<<std::endl;
                // }
                observations.push_back(Hs[i] * real + noise);
            }
        }

        arma::vec get_observation(double t) override
        {
            return observations.at(find_idx(t));
        }

        arma::vec observe(double t, arma::vec state) override
        {
            return Hs.at(find_idx(t)) * state;
        }

        arma::mat observe(double t, arma::mat state) override
        {
            int idx = std::lower_bound(times.begin(), times.end(), t) - times.begin();
            return Hs[idx] * state;
        }

        arma::sp_mat linear(double t, arma::vec state) override
        {
            int idx = std::lower_bound(times.begin(), times.end(), t) - times.begin();
            return Hs[idx];
        }

        arma::mat noise(double t) override
        {
            int idx = std::lower_bound(times.begin(), times.end(), t) - times.begin();
            return noises[idx];
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

        arma::sp_mat generate()
        {
            arma::sp_mat H(obs_num, dim);
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

    double compute_skewness(const arma::mat &ensemble)
    {
        // mean and variance
        int ensemble_size = ensemble.n_cols;
        arma::vec mean = arma::vec(arma::mean(ensemble, 1));
        arma::mat deviation = ensemble.each_col() - mean;
        arma::mat variance = deviation * deviation.t() / ensemble_size;
        arma::mat var_inverse = arma::pinv(variance);

        // calculate skewness
        double skewness = 0;
        for (int i = 0; i < ensemble_size; i++)
        {
            for (int j = 0; j < ensemble_size; j++)
            {
                arma::mat t = deviation.col(i).t() * var_inverse * deviation.col(j);
                skewness += pow(t(0, 0), 3);
            }
        }
        skewness /= ensemble_size * ensemble_size;
        return skewness;
    }

    double compute_kurtosis(const arma::mat &ensemble)
    {
        // mean and variance
        double ensemble_size = ensemble.n_cols;
        double p = ensemble.n_rows;
        arma::vec mean = arma::vec(arma::mean(ensemble, 1));
        arma::mat deviation = ensemble.each_col() - mean;
        arma::mat variance = deviation * deviation.t() / ensemble_size;
        arma::mat var_inverse = arma::pinv(variance);

        // calculate kurtosis
        double kurtosis = 0;
        for (int i = 0; i < ensemble_size; i++)
        {
            arma::mat t = deviation.col(i).t() * var_inverse * deviation.col(i);
            kurtosis += pow(t(0, 0), 2);
        }
        kurtosis /= ensemble_size;

        // convert to N(0, 1)
        kurtosis -= (ensemble_size - 1) / (ensemble_size + 1) * p * (p + 2);
        kurtosis /= sqrt(8.0 / ensemble_size * p * (p + 2));
        return kurtosis;
    }

    void print_singular_values(const arma::mat &var)
    {
        arma::rowvec svd = arma::svd(var).t();
        double sum = arma::accu(svd);

        // std::cout<<"biggest svd: "<<svd[0]<<'\t'<<"smallest svd: "<<svd[svd.n_rows-1]<<std::endl;
        // std::cout<<svd.t()<<std::endl;
        std::cout << arma::cumsum(svd) / sum;
    }

    void print_singular_values(const std::vector<arma::mat> &vars)
    {
        double singular_max = -1;
        double singular_min = std::numeric_limits<double>().max();

        for (const arma::mat &var : vars)
        {
            arma::vec svd = arma::svd(var);
            singular_max = std::max(singular_max, svd[0]);
            singular_min = std::min(singular_min, svd[svd.n_rows - 1]);
        }
        std::cout << "biggest svd: " << singular_max << '\t' << "smallest svd: " << singular_min << std::endl;
    }

    arma::mat compute_bino(arma::drowvec orders, int n)
    {
        arma::mat bino(n + 1, orders.n_cols, arma::fill::none);

        bino.row(0) = arma::drowvec(orders.n_cols, arma::fill::ones);
        // std::cout<<"compute okay\n";
        for (int i = 1; i < n + 1; i++)
        {
            bino.row(i) = (1. - (1. + orders) / i) % bino.row(i - 1);
        }

        return bino;
    }

    arma::vec b_alpha(double alpha, int n)
    {
        arma::vec res(n + 2, arma::fill::zeros);
        for (int i = 1; i < n + 2; i++)
        {
            res[i] = pow(i, 1 - alpha);
        }
        return res.subvec(1, n + 1) - res.subvec(0, n);
    }
}