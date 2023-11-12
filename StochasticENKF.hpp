#pragma once
#include <vector>
#include <utility>
#include <tuple>
#include <future>
#include <iostream>
#include <thread>
#include <memory>
#include <math.h>
#include <limits>

#include "utility.hpp"


namespace shiki{

template<typename T, typename S>
std::vector<vec> 
accumulated_stochastic_ENKF(
    int state_dim, int time_dim, 
    int ensemble_size, int iters_num, mat inflation,
    vec init_average, mat init_uncertainty, 
    std::vector<vec>& ob_results, errors& ob_errors, T ob_op, 
    S model, errors& sys_errors
    ){
    // 初始化
    std::vector<vec> res;
    mat ensemble = arma::mvnrnd(init_average, init_uncertainty, ensemble_size);

    for(int i=0; i<iters_num; i++){
        std::cout<<"time step: "<<i<<"\tn_rows: "<<ensemble.n_rows<<"\tn_cols: "<<ensemble.n_cols;
        std::cout<<"\tstate dimension: "<<state_dim<<"\ttime dimension: "<<time_dim<<"\n";

        if(!ob_results[i].is_empty()){
            int ob_size = ob_results[i].size();
            // 如果这个时刻有观测，则进行同化和修正
            // 生成扰动后的观测
            mat temp, perturb(ob_size, ensemble_size, arma::fill::zeros);
            if(arma::inv(temp, ob_errors[i]))
                perturb = arma::mvnrnd(vec(ob_size, arma::fill::zeros), ob_errors[i], ensemble_size);
            mat after_perturb = perturb.each_col() + ob_results[i];

            // 获取真正参与ENKF的部分
            int last_row;
            if(ensemble.n_rows <= state_dim * time_dim){
                last_row = ensemble.n_rows;
            }else{
                last_row = state_dim * time_dim;
            }

            // 平均值
            mat ensemble_mean = mean(ensemble.submat(0,0,last_row-1,ensemble.n_cols-1), 1);
            mat perturb_mean = mean(perturb, 1);

            // 为了符合算法说明，暂且用下划线
            // 观测后集合
            mat y_f = ob_op(ensemble);
            
            mat x_f = (ensemble.submat(0,0,last_row-1,ensemble.n_cols-1).each_col() - ensemble_mean) / sqrt(ensemble_size - 1);
            mat y_mean = mean(y_f, 1);

            mat auxiliary = after_perturb - y_f;
            temp = y_f - perturb;
            y_f = (temp.each_col() - (y_mean - perturb_mean)) / sqrt(ensemble_size - 1);

            // 计算增益矩阵
            mat H_inflation = ob_op(inflation.submat(0,0,last_row-1,last_row-1));
            temp = y_f * y_f.t() + ob_op(H_inflation.t());
            mat gain = (x_f * y_f.t() + H_inflation.t()) * pinv(temp);
            // 更新集合
            ensemble.submat(0,0,last_row-1,ensemble.n_cols-1) += gain * auxiliary;
        }else{
            ensemble = ensemble;
        }

        // 储存结果
        res.push_back(vec(mean(ensemble, 1)));

        // 如果不是最后一步，就往前推进
        if(i != iters_num-1)
            ensemble = model(ensemble, i, sys_errors[i]);
    }

    return res;
}

template<typename T, typename S>
std::vector<vec> 
accumulated_stochastic_ENKF_inverse(
    int state_dim, int time_dim, int inverse_window,
    int ensemble_size, int iters_num, mat inflation,
    vec init_average, mat init_uncertainty, 
    std::vector<vec>& ob_results, errors& ob_errors, T ob_op, 
    S model, errors& sys_errors
    ){
    // 初始化
    std::vector<vec> res;
    mat ensemble = arma::mvnrnd(init_average, init_uncertainty, ensemble_size);

    for(int i=0; i<iters_num; i++){
        std::cout<<"time step: "<<i<<"\tn_rows: "<<ensemble.n_rows<<"\tn_cols: "<<ensemble.n_cols;
        std::cout<<"\tstate dimension: "<<state_dim<<"\ttime dimension: "<<time_dim<<"\n";

        if(!ob_results[i].is_empty()){
            int ob_size = ob_results[i].size();
            // 如果这个时刻有观测，则进行同化和修正
            // 生成扰动后的观测
            mat temp, perturb(ob_size, ensemble_size, arma::fill::zeros);
            if(arma::inv(temp, ob_errors[i]))
                perturb = arma::mvnrnd(vec(ob_size, arma::fill::zeros), ob_errors[i], ensemble_size);
            mat after_perturb = perturb.each_col() + ob_results[i];

            // 获取真正参与ENKF的部分
            int last_row, first_row;
            if(ensemble.n_rows <= 2 * state_dim * inverse_window){
                last_row = ensemble.n_rows / 2;
                first_row = last_row;
                last_row = min(state_dim * time_dim, last_row);
            }else{
                first_row = ensemble.n_rows - state_dim * inverse_window;
                last_row = min(state_dim * time_dim, first_row);
            }
            mat to_enkf(last_row+ensemble.n_rows-first_row, ensemble.n_cols, arma::fill::none);
            to_enkf.submat(0,0,last_row-1,to_enkf.n_cols-1) = ensemble.submat(0,0,last_row-1,ensemble.n_cols-1);
            to_enkf.submat(last_row,0,to_enkf.n_rows-1,to_enkf.n_cols-1) = ensemble.submat(first_row,0,ensemble.n_rows-1,ensemble.n_cols-1);

            // 平均值
            mat ensemble_mean = mean(to_enkf, 1);
            mat perturb_mean = mean(perturb, 1);

            // 为了符合算法说明，暂且用下划线
            // 观测后集合
            mat y_f = ob_op(ensemble);
            
            mat x_f = (to_enkf.each_col() - ensemble_mean) / sqrt(ensemble_size - 1);
            mat y_mean = mean(y_f, 1);

            mat auxiliary = after_perturb - y_f;
            temp = y_f - perturb;
            y_f = (temp.each_col() - (y_mean - perturb_mean)) / sqrt(ensemble_size - 1);

            // 计算增益矩阵
            mat H_inflation = ob_op(inflation.submat(0,0,to_enkf.n_rows-1,to_enkf.n_rows-1));
            temp = y_f * y_f.t() + ob_op(H_inflation.t());
            mat gain = (x_f * y_f.t() + H_inflation.t()) * pinv(temp);
            // 更新集合
            to_enkf += gain * auxiliary;
            ensemble.submat(0,0,last_row-1,ensemble.n_cols-1) = to_enkf.submat(0,0,last_row-1,to_enkf.n_cols-1);
            ensemble.submat(first_row,0,ensemble.n_rows-1,ensemble.n_cols-1) = to_enkf.submat(last_row,0,to_enkf.n_rows-1,to_enkf.n_cols-1);
        }else{
            ensemble = ensemble;
        }

        // 储存结果
        res.push_back(vec(mean(ensemble, 1)));

        // 如果不是最后一步，就往前推进
        if(i != iters_num-1)
            ensemble = model(ensemble, i, sys_errors[i]);
    }

    return res;
}

template<typename T, typename S>
std::tuple<std::vector<vec>, std::vector<double>, std::vector<double>>  
accumulated_stochastic_ENKF_normal_test(
    int state_dim, int time_dim, 
    int ensemble_size, int iters_num, mat inflation,
    vec init_average, mat init_uncertainty, 
    std::vector<vec>& ob_results, errors& ob_errors, T ob_op, 
    S model, errors& sys_errors
    ){
    // 初始化
    std::vector<vec> res;
    std::vector<double> skewnesses;
    std::vector<double> kurtosises;
    mat ensemble = arma::mvnrnd(init_average, init_uncertainty, ensemble_size);

    for(int i=0; i<iters_num; i++){
        std::cout<<"time step: "<<i<<"\tn_rows: "<<ensemble.n_rows<<"\tn_cols: "<<ensemble.n_cols;
        std::cout<<"\tstate dimension: "<<state_dim<<"\ttime dimension: "<<time_dim<<"\n";

        // 获取真正参与ENKF的部分
        int last_row;
        if(ensemble.n_rows <= state_dim * time_dim){
            last_row = ensemble.n_rows;
        }else{
            last_row = state_dim * time_dim;
        }

        std::future<double> skewness = std::async(std::launch::deferred, compute_skewness, ensemble.submat(0,0,last_row-1,ensemble.n_cols-1));
        std::future<double> kurtosis = std::async(std::launch::deferred, compute_kurtosis, ensemble.submat(0,0,last_row-1,ensemble.n_cols-1));  

        if(!ob_results[i].is_empty()){
            int ob_size = ob_results[i].size();
            // 如果这个时刻有观测，则进行同化和修正
            // 生成扰动后的观测
            mat temp, perturb(ob_size, ensemble_size, arma::fill::zeros);
            if(arma::inv(temp, ob_errors[i]))
                perturb = arma::mvnrnd(vec(ob_size, arma::fill::zeros), ob_errors[i], ensemble_size);
            mat after_perturb = perturb.each_col() + ob_results[i];

            // 平均值
            mat ensemble_mean = mean(ensemble.submat(0,0,last_row-1,ensemble.n_cols-1), 1);
            mat perturb_mean = mean(perturb, 1);

            // 为了符合算法说明，暂且用下划线
            // 观测后集合
            mat y_f = ob_op(ensemble);
            
            mat x_f = (ensemble.submat(0,0,last_row-1,ensemble.n_cols-1).each_col() - ensemble_mean) / sqrt(ensemble_size - 1);
            mat y_mean = mean(y_f, 1);

            mat auxiliary = after_perturb - y_f;
            temp = y_f - perturb;
            y_f = (temp.each_col() - (y_mean - perturb_mean)) / sqrt(ensemble_size - 1);

            // 计算增益矩阵
            mat H_inflation = ob_op(inflation.submat(0,0,last_row-1,last_row-1));
            temp = y_f * y_f.t() + ob_op(H_inflation.t());
            mat gain = (x_f * y_f.t() + H_inflation.t()) * pinv(temp);
            // 更新集合
            ensemble.submat(0,0,last_row-1,ensemble.n_cols-1) += gain * auxiliary;
        }else{
            ensemble = ensemble;
        }

        // 储存结果
        res.push_back(vec(mean(ensemble, 1)));

        // 如果不是最后一步，就往前推进
        if(i != iters_num-1)
            ensemble = model(ensemble, i, sys_errors[i]);

        skewnesses.push_back(skewness.get());
        kurtosises.push_back(kurtosis.get());
    }

    return std::tuple<std::vector<vec>, std::vector<double>, std::vector<double>>(res, skewnesses, kurtosises);
}

arma::vec b_alpha(double alpha, int n){
    vec res(n+2, arma::fill::zeros);
    for(int i=1; i<n+2; i++){
        res[i] = pow(i, 1-alpha);
    }
    return res.subvec(1, n+1) - res.subvec(0, n);
}

arma::mat compute_bino(arma::drowvec orders, int n){
    arma::mat bino(n+1, orders.n_cols, arma::fill::none);
    
    bino.row(0) = drowvec(orders.n_cols, arma::fill::ones);
    // std::cout<<"compute okay\n";
    for(int i=1; i<n+1; i++){
        bino.row(i) = (1. - (1. + orders) / i ) % bino.row(i-1);
    }

    return bino;
}


}