#pragma once
#include <armadillo>
#include <utility>
#include <exception>
#include <vector>
#include "StochasticENKF.hpp"

using arma::vec;
using arma::mat;

extern std::pair<vec, mat> compute_mean_variance(const mat& ensemble, const vec& weights);


template<typename Model, typename Ob>
std::vector< std::pair<mat, vec> >
ParticleFilter(mat ensemble, Model model, Ob likehood, std::vector<vec> obResults, Errors sys_vars){
    int num_iter = obResults.size();
    int ensemble_size = ensemble.n_cols;
    std::vector< std::pair<mat, vec> > results{};
    vec weights(ensemble_size, arma::fill::value(1./ensemble_size));

    for(int i=0; i<num_iter; i++){
        if(obResults[i].empty()){
            // 空的观测，不更新权重，只是推进模型
            results.push_back({ensemble, weights});
        }else{
            // 有观测，需要更新权重
            double sum = 0;
            for(int j=0; j<ensemble_size; j++){
                weights[j] = weights[j] * likehood(ensemble.col(j), obResults[i]);
                sum += weights[j];
            }
            weights /= sum;
            results.push_back({ensemble, weights});

            auto mean_var = compute_mean_variance(ensemble, weights);
            ensemble = arma::mvnrnd(mean_var.first, mean_var.second, ensemble_size);
            weights = vec(ensemble_size, arma::fill::value(1./ensemble_size));
        }

        ensemble = model(ensemble, i, sys_vars[i]);
    }

    return results;
}

std::pair<vec, mat> compute_mean_variance(const mat& ensemble, const vec& weights){
    vec mean(ensemble.n_rows, arma::fill::zeros);
    for(int i=0; i<ensemble.n_cols; i++){
        mean += weights(i) * ensemble.col(i);
    }

    mat var(ensemble.n_rows, ensemble.n_rows, arma::fill::zeros);
    for(int j=0; j<ensemble.n_cols; j++){
        vec devication = ensemble.col(j) - mean;
        var += weights(j) * devication * devication.t();
    }
    
    return std::pair{mean, var};
}

