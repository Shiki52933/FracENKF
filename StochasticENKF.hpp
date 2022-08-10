#include <armadillo>
#include <vector>
#include <memory>
#include <math.h>

using namespace arma;
typedef arma::mat mat;

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

std::vector<vec> StochasticENKF(int ensembleSize, vec initAverage, mat initUncertainty, std::vector<vec>& obResults, 
                                int numIters, Errors& obErrors, ObserveOperator obOp, Model model, Errors& sysErrors){
    // 初始化
    std::vector<vec> res;
    mat ensemble = mvnrnd(initAverage, initUncertainty, ensembleSize);

    for(int i=0; i<numIters; i++){
        mat ensembleAnalysis;

        if(!obResults[i].is_empty()){
            int obSize = obResults[i].size();
            // 如果这个时刻有观测，则进行同化和修正
            // 生成扰动后的观测
            mat perturb = mvnrnd(vec(obSize, arma::fill::zeros), obErrors[i], ensembleSize);
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
            mat temp = y_f - perturb;
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
        
    }

    return res;
}

vec BAlpha(double alpha, int n){
    vec res(n+2, arma::fill::zeros);
    for(int i=1; i<n+2; i++){
        res[i] = pow(i, 1-alpha);
    }
    return res.subvec(1, n+1) - res.subvec(0, n);
}