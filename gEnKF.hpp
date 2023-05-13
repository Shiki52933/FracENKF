#include "fEKF.hpp"

namespace shiki{

template<typename S>
std::vector<vec>
simple_group_ENKF(
    int iters_num, double h, 
    mat ensemble, std::vector<sp_mat> vars, 
    std::vector<vec>& ob_results, mat H, errors& ob_errors,
    S model, linearize model_linear, errors& sys_errors,
    double inflation=1.1
    ){
    // 初始化
    int ensemble_size = ensemble.n_cols;
    mat I = arma::eye(ensemble.n_rows, ensemble.n_rows);
    std::vector<vec> res;

    for(int i=0; i<iters_num; i++){
        // std::cout<<"time step: "<<i<<"\tn_rows: "<<ensemble.n_rows<<"\tn_cols: "<<ensemble.n_cols<<'\n';

        if(!ob_results[i].is_empty()){
            // 平均值
            mat ensemble_mean = mean(ensemble, 1);
            mat x_f = (ensemble.each_col() - ensemble_mean) / sqrt(ensemble_size - 1);
            mat P_f = x_f * x_f.t();

            // Var_i
            sp_mat ave_var = vars[0];
            for(int j=1; j<vars.size(); ++j){
                ave_var += vars[j];
            }
            ave_var /= vars.size();
            P_f += ave_var;

            // 计算增益矩阵
            mat gain = P_f * H.t() * inv(H * P_f * H.t() + ob_errors[i]);
            mat ob_f = H * ensemble;

            // 分析步
            ensemble += gain * (ob_results[i] - ob_f.each_col());
            // 更新vars
            mat coef = I - gain * H;
            mat change = gain * ob_errors[i] * gain.t();
            for(auto &var: vars){
                var = coef * var * coef.t() + change;
            }
        }

        // 储存结果
        res.push_back(vec(mean(ensemble, 1)));

        // 如果不是最后一步，就往前推进
        if(i != iters_num-1){
            for(int j=0; j<ensemble_size; ++j){
                mat D = model_linear(ensemble.col(j));
                // 是否将sys_error放在这有待商榷
                // 个人认为不要放在这
                // vars[j] += h * h * (D * vars[j] * D.t() + sys_errors[i]);
                // vars[j] += h * h * (D * vars[j] * D.t());
                vars[j] = (I + h * D) * vars[j] * (I + h * D).t() + sys_errors[i];
            }
            ensemble = model(ensemble, i, 0*sys_errors[i]);
            // ensemble = model(ensemble, i, sys_errors[i]);
        }
    }

    return res;
}


template<typename S>
std::vector<vec>
group_ENKF(
    int iters_num, double h,
    mat ensemble, std::vector<mat> vars, 
    std::vector<vec>& ob_results, mat H, errors& ob_errors,
    S model, linearize model_linear, errors& sys_errors
    ){
    // 初始化
    int ensemble_size = ensemble.n_cols;
    mat I = arma::eye(ensemble.n_rows, ensemble.n_rows);
    std::vector<vec> res;

    for(int i=0; i<iters_num; i++){
        // std::cout<<"time step: "<<i<<"\tn_rows: "<<ensemble.n_rows<<"\tn_cols: "<<ensemble.n_cols<<'\n';

        if(!ob_results[i].is_empty()){
            // 平均值
            mat ensemble_mean = mean(ensemble, 1);
            mat x_f = (ensemble.each_col() - ensemble_mean) / sqrt(ensemble_size - 1);
            mat P_f = x_f * x_f.t();

            // Var_i
            mat ave_var = vars[0];
            for(int j=1; j<vars.size(); ++j){
                ave_var += vars[j];
            }
            ave_var /= vars.size();
            P_f += ave_var;

            // 计算增益矩阵
            mat gain = P_f * H.t() * inv(H * P_f * H.t() + ob_errors[i]);
            mat ob_f = H * ensemble;

            // 分析步
            ensemble += gain * (ob_results[i] - ob_f.each_col());
            // 更新vars
            mat coef = I - gain * H;
            mat change = gain * ob_errors[i] * gain.t();
            for(auto &var: vars){
                var = coef * var * coef.t() + change;
            }
        }

        // 储存结果
        res.push_back(vec(mean(ensemble, 1)));

        // 如果不是最后一步，就往前推进
        if(i != iters_num-1){
            for(int j=0; j<ensemble_size; ++j){
                mat D = model_linear(ensemble.col(j));
                // 是否将sys_error放在这有待商榷
                // 个人认为不要放在这
                // vars[j] += h * h * (D * vars[j] * D.t() + sys_errors[i]);
                // vars[j] += h * h * (D * vars[j] * D.t());
                vars[j] = (I + h * D) * vars[j] * (I + h * D).t() + sys_errors[i];
            }
            ensemble = model(ensemble, i, 0*sys_errors[i]);
            // ensemble = model(ensemble, i, sys_errors[i]);
        }
    }

    return res;
}

}