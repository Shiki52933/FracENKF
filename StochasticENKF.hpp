#pragma once
#include <armadillo>
#include <vector>
#include <utility>
#include <tuple>
#include <future>
#include <iostream>
#include <thread>
#include <memory>
#include <math.h>
#include <limits>

#include "Krylov.hpp"


namespace shiki{

using arma::vec;
using arma::drowvec;
using arma::mat;
using arma::sp_mat;

class errors{
    std::vector<std::shared_ptr<mat>> m_ptrs;

public:
    void add(std::shared_ptr<mat> ptr2mat){
        this->m_ptrs.push_back(ptr2mat);
    }

    // 返回误差矩阵，不做边界检查
    mat& operator[](int idx){
        return *m_ptrs[idx];
    }
}; 

typedef mat (*observe_operator)(mat&);
typedef mat (*dynamic_model)(mat& ensembleAnalysis, int idx, mat& sysVar);
extern double compute_skewness(const mat& ensemble);
extern double compute_kurtosis(const mat& ensemble);

template<typename T>
inline static T min(T a, T b){
    return a < b ? a : b;
}

template<typename T, typename S>
std::tuple<std::vector<vec>, std::vector<double>, std::vector<double>> 
stochastic_ENKF_normal_test(
    int ensemble_size, int iters_num,
    vec init_average, mat init_uncertainty, 
    std::vector<vec>& ob_results, T ob_op, errors& ob_errors,
    S model, errors& sys_errors
    ){
    // 初始化
    std::vector<vec> res;
    std::vector<double> skewnesses;
    std::vector<double> kurtosises;
    mat ensemble = arma::mvnrnd(init_average, init_uncertainty, ensemble_size);

    for(int i=0; i<iters_num; i++){
        std::cout<<"time step: "<<i<<"\tn_rows: "<<ensemble.n_rows<<"\tn_cols: "<<ensemble.n_cols<<'\n';
        mat ensemble_analysis;

        std::future<double> skewness = std::async(std::launch::deferred, compute_skewness, ensemble);
        // skewnesses.push_back(compute_skewness(ensemble));
        std::future<double> kurtosis = std::async(std::launch::deferred, compute_kurtosis, ensemble);  
        // kurtosises.push_back(compute_kurtosis(ensemble));

        if(!ob_results[i].is_empty()){
            int ob_size = ob_results[i].size();
            // 如果这个时刻有观测，则进行同化和修正
            // 生成扰动后的观测
            mat temp, perturb(ob_size, ensemble_size, arma::fill::zeros);
            if(arma::inv(temp, ob_errors[i]))
                perturb = arma::mvnrnd(vec(ob_size, arma::fill::zeros), ob_errors[i], ensemble_size);
            mat after_perturb = perturb.each_col() + ob_results[i];

            // 平均值
            mat ensemble_mean = mean(ensemble, 1);
            mat perturb_mean = mean(perturb, 1);

            // 观测后集合
            mat y_f = ob_op(ensemble);
            
            mat x_f = (ensemble.each_col() - ensemble_mean) / sqrt(ensemble_size - 1);
            mat y_mean = mean(y_f, 1);

            mat auxiliary = after_perturb - y_f;
            temp = y_f - perturb;
            y_f = (temp.each_col() - (y_mean - perturb_mean)) / sqrt(ensemble_size - 1);

            // 计算增益矩阵
            mat gain = x_f * y_f.t() * inv(y_f * y_f.t());
            // 更新集合
            ensemble_analysis = ensemble + gain * auxiliary;
        }else{
            ensemble_analysis = ensemble;
        }

        // 储存结果
        res.push_back(vec(mean(ensemble_analysis, 1)));

        // 如果不是最后一步，就往前推进
        if(i != iters_num-1)
            ensemble = model(ensemble_analysis, i, sys_errors[i]);

        skewnesses.push_back(skewness.get());
        kurtosises.push_back(kurtosis.get());
    }

    return std::tuple<std::vector<vec>, std::vector<double>, std::vector<double>>(res, skewnesses, kurtosises);
}

template<typename T, typename S>
std::vector<vec>
stochastic_ENKF(
    int ensemble_size, int iters_num,
    vec init_average, mat init_uncertainty, 
    std::vector<vec>& ob_results, T ob_op, errors& ob_errors,
    S model, errors& sys_errors
    ){
    // 初始化
    std::vector<vec> res;
    mat ensemble = arma::mvnrnd(init_average, init_uncertainty, ensemble_size);

    for(int i=0; i<iters_num; i++){
        // std::cout<<"time step: "<<i<<"\tn_rows: "<<ensemble.n_rows<<"\tn_cols: "<<ensemble.n_cols<<'\n';
        mat ensemble_analysis;

        if(!ob_results[i].is_empty()){
            int ob_size = ob_results[i].size();
            // 如果这个时刻有观测，则进行同化和修正
            // 生成扰动后的观测
            mat temp, perturb(ob_size, ensemble_size, arma::fill::zeros);
            if(arma::inv(temp, ob_errors[i]))
                perturb = arma::mvnrnd(vec(ob_size, arma::fill::zeros), ob_errors[i], ensemble_size);
            mat after_perturb = perturb.each_col() + ob_results[i];

            // 平均值
            mat ensemble_mean = mean(ensemble, 1);
            mat perturb_mean = mean(perturb, 1);

            // 为了符合算法说明，暂且用下划线
            // 观测后集合
            mat y_f = ob_op(ensemble);
            
            mat x_f = (ensemble.each_col() - ensemble_mean) / sqrt(ensemble_size - 1);
            mat y_mean = mean(y_f, 1);

            mat auxiliary = after_perturb - y_f;
            temp = y_f - perturb;
            y_f = (temp.each_col() - (y_mean - perturb_mean)) / sqrt(ensemble_size - 1);

            // 计算增益矩阵
            mat gain = x_f * y_f.t() * inv(y_f * y_f.t());
            // 更新集合
            ensemble_analysis = ensemble + gain * auxiliary;
        }else{
            ensemble_analysis = ensemble;
        }

        // 储存结果
        res.push_back(vec(mean(ensemble_analysis, 1)));

        // 如果不是最后一步，就往前推进
        if(i != iters_num-1)
            ensemble = model(ensemble_analysis, i, sys_errors[i]);
    }

    return res;
}

template<typename T>
std::vector<vec>
stochastic_ENKF(
    int ensemble_size, int iters_num,
    vec init_average, mat init_uncertainty, 
    std::vector<vec>& ob_results, mat ob_op, errors& ob_errors,
    T model, errors& sys_errors
    ){
    // 初始化
    std::vector<vec> res;
    mat ensemble = arma::mvnrnd(init_average, init_uncertainty, ensemble_size);

    for(int i=0; i<iters_num; i++){
        // std::cout<<"time step: "<<i<<"\tn_rows: "<<ensemble.n_rows<<"\tn_cols: "<<ensemble.n_cols<<'\n';
        mat ensemble_analysis;

        if(!ob_results[i].is_empty()){
            int ob_size = ob_results[i].size();
            // 如果这个时刻有观测，则进行同化和修正
            // 生成扰动后的观测
            mat temp, perturb(ob_size, ensemble_size, arma::fill::zeros);
            if(arma::inv(temp, ob_errors[i]))
                perturb = arma::mvnrnd(vec(ob_size, arma::fill::zeros), ob_errors[i], ensemble_size);
            mat after_perturb = perturb.each_col() + ob_results[i];

            // 为了符合算法说明，暂且用下划线
            // 观测后集合
            mat ensemble_mean = mean(ensemble, 1);
            mat x_f = (ensemble.each_col() - ensemble_mean) / sqrt(ensemble_size - 1);
            mat P_f = x_f * x_f.t();

            mat y_f = ob_op * ensemble;
            mat auxiliary = after_perturb - y_f;

            // 计算增益矩阵
            mat gain = P_f * ob_op.t() * inv(ob_op * P_f * ob_op.t() + ob_errors[i]);
            // 更新集合
            ensemble_analysis = ensemble + gain * auxiliary;
        }else{
            ensemble_analysis = ensemble;
        }

        // 储存结果
        res.push_back(vec(mean(ensemble_analysis, 1)));

        // 如果不是最后一步，就往前推进
        if(i != iters_num-1)
            ensemble = model(ensemble_analysis, i, sys_errors[i]);
    }

    return res;
}

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

vec b_alpha(double alpha, int n){
    vec res(n+2, arma::fill::zeros);
    for(int i=1; i<n+2; i++){
        res[i] = pow(i, 1-alpha);
    }
    return res.subvec(1, n+1) - res.subvec(0, n);
}

double compute_skewness(const mat& ensemble){
    // mean and variance
    int ensemble_size = ensemble.n_cols;
    vec mean = vec(arma::mean(ensemble, 1));
    mat x_f = (ensemble.each_col() - mean) / sqrt(ensemble_size);
    mat variance = x_f *x_f.t();
    mat var_inverse = arma::pinv(variance);

    // calculate skewness
    double skewness = 0;
    for(int i=0; i<ensemble_size; i++){
        mat deviation_i = ensemble.col(i) - mean;
        for(int j=0; j<ensemble_size; j++){
            mat deviation_j = ensemble.col(j) - mean;
            mat t = deviation_i.t() * var_inverse * deviation_j;
            skewness += pow(t(0,0), 3);
        }
    }
    skewness /= ensemble_size * ensemble_size;
    return skewness;
}

double compute_kurtosis(const mat& ensemble){
    // mean and variance
    double ensemble_size = ensemble.n_cols;
    double p = ensemble.n_rows;
    vec mean = vec(arma::mean(ensemble, 1));
    mat x_f = (ensemble.each_col() - mean) / sqrt(ensemble_size);
    mat variance = x_f *x_f.t();
    mat var_inverse = arma::pinv(variance);

    // calculate kurtosis
    double kurtosis = 0;
    for(int i=0; i<ensemble_size; i++){
        mat deviation_i = ensemble.col(i) - mean;
        mat t = deviation_i.t() * var_inverse * deviation_i;
        kurtosis += pow(t(0,0), 2);
    }
    kurtosis /= ensemble_size;

    // convert to N(0, 1)
    kurtosis -= (ensemble_size - 1) / (ensemble_size + 1) * p * (p + 2);
    kurtosis /= sqrt(8.0 / ensemble_size * p * (p + 2));
    return kurtosis;
}

mat compute_bino(drowvec orders, int n){
    mat bino(n+1, orders.n_cols, arma::fill::none);
    
    bino.row(0) = drowvec(orders.n_cols, arma::fill::ones);
    // std::cout<<"compute okay\n";
    for(int i=1; i<n+1; i++){
        bino.row(i) = (1. - (1. + orders) / i ) % bino.row(i-1);
    }

    return bino;
}


void print_singular_values(const mat &var){
    arma::rowvec svd = arma::svd(var).t();
    double sum = arma::accu(svd);
        
    // std::cout<<"biggest svd: "<<svd[0]<<'\t'<<"smallest svd: "<<svd[svd.n_rows-1]<<std::endl;
    // std::cout<<svd.t()<<std::endl;
    std::cout<<arma::cumsum(svd)/sum;

}

void print_singular_values(const std::vector<mat> &vars){
    double singular_max = -1;
    double singular_min = std::numeric_limits<double>().max();

    for(const mat &var: vars){
        vec svd = arma::svd(var);
        singular_max = std::max(singular_max, svd[0]);
        singular_min = std::min(singular_min, svd[svd.n_rows-1]);
    }
    std::cout<<"biggest svd: "<<singular_max<<'\t'<<"smallest svd: "<<singular_min<<std::endl;
}



}