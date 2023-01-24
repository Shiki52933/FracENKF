#include "StochasticENKF.hpp"
#include <vector>
#include <memory>

namespace shiki{

typedef mat (*linearize)(vec);

template<typename T>
mat fEKF(
    int state_dim, drowvec orders, double h,
    vec init_mean, mat init_var, 
    std::vector<vec> ob_list, const mat& H, errors ob_errs, 
    T model, linearize modelLinear, errors sys_vars
    ){
    int iter_num = ob_list.size();

    // 保存同化结果和协方差矩阵
    mat result(iter_num, state_dim, arma::fill::none);
    result.row(0) = init_mean.t();
    std::vector<mat> former_vars;
    former_vars.push_back(init_var);

    // 计算gamma
    mat bino = compute_bino(orders, iter_num+1);
    std::vector<sp_mat> gammas;
    for(int i=0; i<=iter_num+1; i++){
        sp_mat gamma_i(state_dim, state_dim);
        for(int j=0; j<state_dim; j++){
            gamma_i(j,j) = bino(i,j);
        }
        gammas.push_back(gamma_i);
    }

    // 计算缩放系数
    mat mat_h(state_dim, state_dim, arma::fill::zeros);
    for(int i=0; i<state_dim; i++)
        mat_h(i,i) = pow(h, orders(i));

    for(int i=0; i<iter_num; i++){
        // 循环开始时，我们已经知道预测值和预测方差，
        // 我们将在按模型推进后维护预测值和预测方差
        if(!ob_list[i].is_empty()){
            // 非空的观测，需要同化
            vec mean = result.row(i).t();
            mat var = former_vars[i];

            mat gain = var * H.t() * arma::inv(H * var * H.t() + ob_errs[i]);
            mean += gain * (ob_list[i] - H * mean);
            var = (arma::eye(gain.n_rows, gain.n_rows) - gain * H) * var;

            result.row(i) = mean.t();
            former_vars[i] = var;
        }

        // 推进
        mat temp = result.submat(0,0,i,result.n_cols-1);
        temp = arma::reverse(temp).t();
        vec reshaped = arma::reshape(temp, temp.n_rows*temp.n_cols, 1);

        vec mean = model(reshaped, i, 0*sys_vars[i]);
        mean = mean.subvec(0, state_dim-1);
        if(i != iter_num-1)
            result.row(i+1) = mean.t();

        // 计算预测协方差阵
        mat F = modelLinear(result.row(i).t());
        F = mat_h * F;
        const mat& P_f = former_vars.back();
        mat new_P = F * P_f * F.t() + gammas[1] * P_f * F.t() + 
            F * P_f * gammas[1].t() + mat_h * sys_vars[i] * mat_h.t();
        for(int j=1; j<=i+1; j++){
            new_P += gammas[j] * former_vars[i+1-j] * gammas[j].t();
        }
        if(i != iter_num-1)
            former_vars.push_back(new_P);
    }

    return result;
}

}
