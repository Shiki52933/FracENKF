#pragma once
#include <armadillo>
#include <vector>
#include "StochasticENKF.hpp"

using arma::vec;
using arma::mat;

template<typename T>
std::vector<vec> 
ThreeDVar(int state_num, int time_window, 
    vec mean, mat variance, 
    mat ob_op, std::vector<vec> ob_lists, Errors ob_vars, 
    T model, Errors sys_vars){
    std::vector<vec> results;
    int iter_num = ob_lists.size();

    for(int i=0; i<iter_num; i++){
        if(!ob_lists[i].is_empty()){
            int num_row = state_num*time_window > mean.n_rows ? mean.n_rows : state_num*time_window;
            // std::cout<<"num_row: "<<num_row<<"\n";

            mat real_ob_op(ob_op.n_rows, num_row, arma::fill::zeros);
            real_ob_op.submat(0,0,ob_op.n_rows-1,ob_op.n_cols-1) = ob_op;
            // std::cout<<"num_row1: "<<num_row<<"\n";

            mat real_variance = variance.submat(0,0,num_row-1, num_row-1);
            mat gain = real_variance * real_ob_op.t() * pinv(real_ob_op * real_variance * real_ob_op.t() + ob_vars[i]);
            // std::cout<<"num_row2: "<<num_row<<"\n";

            vec innovation = gain * (ob_lists[i] - real_ob_op * mean.subvec(0, num_row-1));
            mean.subvec(0, num_row-1) += innovation;
        }

        results.push_back(mean);
        mean = model(mean, i, sys_vars[i]);
    }

    return results;
}