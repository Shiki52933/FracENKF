#pragma once
#include <armadillo>
#include <utility>
#include <vector>

using arma::vec;
using arma::mat;


template<typename Model, typename Ob>
std::vector< std::pair<mat, vec> >
ParticleFilter(mat ensemble, Model model, Ob likehood, std::vector<vec> obResults){
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
                weights[j] = weights[j] * likehood(ensemble[j], obResults[j]);
                sum += weights[j];
            }
            weights /= sum;
            results.push_back({ensemble, weights});
        }

        ensemble = model(ensemble);
    }

    return results;
}
