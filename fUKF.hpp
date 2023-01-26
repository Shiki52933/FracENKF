#include "StochasticENKF.hpp"

namespace shiki{

template<typename T>
mat fUKF(
    double alpha, double beta, double k,
    int state_dim, drowvec orders, double h,
    vec init_mean, mat init_var,
    std::vector<vec> ob_list, const mat& H, errors ob_errs,
    T rhs, errors sys_vars, mat inflation
){
    double lambda = alpha * alpha * (state_dim + k) - state_dim;
    // if(lambda < 0)
    //     throw std::runtime_error("bad coefficient");
    int iter_num = ob_list.size();

    mat result(iter_num, state_dim, arma::fill::none);
    result.row(0) = init_mean.t();
    std::vector<mat> former_vars;
    former_vars.push_back(init_var);

    // 计算缩放系数
    mat mat_h(state_dim, state_dim, arma::fill::zeros);
    for(int i=0; i<state_dim; i++)
        mat_h(i,i) = pow(h, orders(i));
    
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

    for(int i=0; i<iter_num; i++){
        std::cout<<"time step: "<<i<<"\n";
        
        if(!ob_list[i].is_empty()){
            // 更新均值和方差
            vec mean = result.row(i).t();
            mat var = former_vars[i] + inflation;

            mat gain = var * H.t() * arma::inv(H * var * H.t() + ob_errs[i]);
            std::cout<<"error in observation: "<<arma::norm(ob_list[i] - H * mean)<<"\n";
            vec innovation = gain * (ob_list[i] - H * mean);
            mean += innovation;
            var = (arma::eye(gain.n_rows, gain.n_rows) - gain * H) * var;

            result.row(i) = mean.t();
            former_vars[i] = var;
        }

        // 推进

        // 首先取sigma点
        mat temp = (state_dim+lambda) * former_vars[i];
        std::cout<<"trace of variance: "<<arma::trace(temp)<<"\n";
        mat deviations = arma::chol(temp, "lower");

        mat ensemble(state_dim, 2*state_dim+1, arma::fill::none);
        ensemble.col(0) = result.row(i).t();
        ensemble.submat(0,1,state_dim-1,state_dim) = result.row(i).t() + deviations.each_col();
        ensemble.submat(0,state_dim+1, state_dim-1,2*state_dim) = result.row(i).t() - deviations.each_col();
        // 然后计算权重
        vec mean_weight(2*state_dim+1, arma::fill::value( 1./2./(state_dim+lambda) ));
        vec var_weight(2*state_dim+1, arma::fill::value( 1./2./(state_dim+lambda) ));
        mean_weight(0) = lambda / (state_dim + lambda);
        var_weight(0) = lambda / (state_dim + lambda) + 1. - alpha * alpha + beta;
        // 然后通过模型,模型不要噪声
        mat prediction = rhs(ensemble);
        prediction = mat_h * prediction;
        prediction.each_col([&](vec& col){col -= gammas[1] * result.row(i).t();}); 

        // 然后计算方差和协方差
        vec new_mean(state_dim, arma::fill::zeros);
        for(int j=0; j<2*state_dim+1; j++)
            new_mean += mean_weight(j) * prediction.col(j);

        mat new_var(state_dim, state_dim, arma::fill::zeros);
        for(int j=0; j<2*state_dim+1; j++){
            new_var += var_weight(j) * (prediction.col(j) - new_mean) * (prediction.col(j) - new_mean).t();
        }
        new_var += mat_h * sys_vars[i] * mat_h.t();
        // std::cout<<"trace of new variance: "<<arma::trace(new_var)<<"\n";

        // 然后计算均值和方差
        vec real_new_mean = new_mean;
        for(int j=2; j<=i+1; j++)
            real_new_mean -= gammas[j] * (result.row(i+1-j).t());

        mat real_new_var = new_var;
        for(int j=2; j<=i+1; j++){
            real_new_var += gammas[j] * former_vars[i+1-j] * gammas[j].t();
        }

        if(i != iter_num-1){
            result.row(i+1) = real_new_mean.t();
            former_vars.push_back(real_new_var);
        }
    }

    return result;
}

}