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

using arma::mat;
using arma::vec;

class Errors{
    std::vector<std::shared_ptr<mat>> mPtrs;

public:
    void add(std::shared_ptr<mat> ptrMat){
        this->mPtrs.push_back(ptrMat);
    }

    // 返回误差矩阵，不做边界检查
    mat& operator[](int idx){
        return *mPtrs[idx];
    }
}; 

typedef mat (*ObserveOperator)(mat&);
typedef mat (*Model)(mat& ensembleAnalysis, int idx, mat& sysVar);
extern double compute_skewness(const mat& ensemble);
extern double compute_kurtosis(const mat& ensemble);

template<typename T, typename S>
std::tuple<std::vector<vec>, std::vector<double>, std::vector<double>> 
StochasticENKF(int ensembleSize, vec initAverage, mat initUncertainty, std::vector<vec>& obResults, 
               int numIters, Errors& obErrors, T obOp, S model, Errors& sysErrors){
    // 初始化
    std::vector<vec> res;
    std::vector<double> skewnesses;
    std::vector<double> kurtosises;
    mat ensemble = arma::mvnrnd(initAverage, initUncertainty, ensembleSize);

    for(int i=0; i<numIters; i++){
        std::cout<<"time step: "<<i<<'\n';
        mat ensembleAnalysis;

        std::future<double> skewness = std::async(std::launch::deferred, compute_skewness, ensemble);
        // skewnesses.push_back(compute_skewness(ensemble));
        std::future<double> kurtosis = std::async(std::launch::deferred, compute_kurtosis, ensemble);  
        // kurtosises.push_back(compute_kurtosis(ensemble));

        if(!obResults[i].is_empty()){
            int obSize = obResults[i].size();
            // 如果这个时刻有观测，则进行同化和修正
            // 生成扰动后的观测
            mat temp, perturb(obSize, ensembleSize, arma::fill::zeros);
            if(arma::inv(temp, obErrors[i]))
                perturb = arma::mvnrnd(vec(obSize, arma::fill::zeros), obErrors[i], ensembleSize);
            mat afterPerturb = perturb.each_col() + obResults[i];

            // 平均值
            mat ensembleMean = mean(ensemble, 1);
            mat perturbMean = mean(perturb, 1);

            // 为了符合算法说明，暂且用下划线
            // 观测后集合
            mat y_f = obOp(ensemble);
            
            mat x_f = (ensemble.each_col() - ensembleMean) / sqrt(ensembleSize - 1);
            mat y_mean = mean(y_f, 1);

            mat auxiliary = afterPerturb - y_f;
            temp = y_f - perturb;
            y_f = (temp.each_col() - (y_mean - perturbMean)) / sqrt(ensembleSize - 1);

            // 计算增益矩阵
            mat gain = x_f * y_f.t() * (y_f * y_f.t()).i();
            // 更新集合
            ensembleAnalysis = ensemble + gain * auxiliary;
        }else{
            ensembleAnalysis = ensemble;
        }

        // 储存结果
        res.push_back(vec(mean(ensembleAnalysis, 1)));

        // 如果不是最后一步，就往前推进
        if(i != numIters-1)
            ensemble = model(ensembleAnalysis, i, sysErrors[i]);

        skewnesses.push_back(skewness.get());
        kurtosises.push_back(kurtosis.get());
    }

    return std::tuple<std::vector<vec>, std::vector<double>, std::vector<double>>(res, skewnesses, kurtosises);
}

vec BAlpha(double alpha, int n){
    vec res(n+2, arma::fill::zeros);
    for(int i=1; i<n+2; i++){
        res[i] = pow(i, 1-alpha);
    }
    return res.subvec(1, n+1) - res.subvec(0, n);
}

double compute_skewness(const mat& ensemble){
    // mean and variance
    int ensembleSize = ensemble.n_cols;
    vec mean = vec(arma::mean(ensemble, 1));
    mat x_f = (ensemble.each_col() - mean) / sqrt(ensembleSize);
    mat variance = x_f *x_f.t();
    mat var_inverse = variance.i();

    // calculate skewness
    double skewness = 0;
    for(int i=0; i<ensembleSize; i++){
        mat deviation_i = ensemble.col(i) - mean;
        for(int j=0; j<ensembleSize; j++){
            mat deviation_j = ensemble.col(j) - mean;
            mat t = deviation_i.t() * var_inverse * deviation_j;
            skewness += pow(t(0,0), 3);
        }
    }
    skewness /= ensembleSize * ensembleSize;
    return skewness;
}

double compute_kurtosis(const mat& ensemble){
    // mean and variance
    double ensembleSize = ensemble.n_cols;
    double p = ensemble.n_rows;
    vec mean = vec(arma::mean(ensemble, 1));
    mat x_f = (ensemble.each_col() - mean) / sqrt(ensembleSize);
    mat variance = x_f *x_f.t();
    mat var_inverse = variance.i();

    // calculate kurtosis
    double kurtosis = 0;
    for(int i=0; i<ensembleSize; i++){
        mat deviation_i = ensemble.col(i) - mean;
        mat t = deviation_i.t() * var_inverse * deviation_i;
        kurtosis += pow(t(0,0), 2);
    }
    kurtosis /= ensembleSize;

    // convert to N(0, 1)
    kurtosis -= (ensembleSize - 1) / (ensembleSize + 1) * p * (p + 2);
    kurtosis /= sqrt(8.0 / ensembleSize * p * (p + 2));
    return kurtosis;
}