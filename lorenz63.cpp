#include "StochasticENKF.hpp"
#include <armadillo>
#include <iostream>


mat generate_lorenz63(double sigma, double rho, double beta, double dt, double t_max, vec x0, mat sys_var){
    int num_iter = t_max / dt;
    if( dt * num_iter < t_max )
        num_iter++;

    mat sol(3, num_iter+1);
    mat perturb = mvnrnd(vec(3,arma::fill::zeros), sys_var, num_iter);

    sol.col(0) = x0;
    for(int i=0; i<num_iter; i++){
        double x_old = sol.col(i)[0];
        double y_old = sol.col(i)[1];
        double z_old = sol.col(i)[2];
        double dt_ = (i+1)*dt <= t_max ? dt : t_max - (i+1)*dt;

        double x_new = sigma * (y_old - x_old) * dt_ + x_old;
        double y_new = ( x_old * (rho - z_old) - y_old ) * dt_ + y_old;
        double z_new = ( x_old * y_old - beta * z_old ) * dt_ + z_old;

        sol.col(i+1)[0] = x_new;
        sol.col(i+1)[1] = y_new;
        sol.col(i+1)[2] = z_new;
        sol.col(i+1) += perturb.col(i);
    } 
    mat sol_t = sol.t();
    sol_t.save("lorenz63.csv", arma::raw_ascii);
    return sol;
}

mat H_ob(mat& ensemble){
    vec a{0, 1, 1};
    vec b{1, 1, 0};
    mat H(3, 2);
    H.col(0) = a;
    H.col(1) = b;
    H = H.t();
    return H * ensemble;
}

mat model(mat& ensemble, int idx, mat& sys_var){
    double sigma = 10, rho = 28, beta = 8.0/3;
    double dt = 0.001;

    mat sol = ensemble;
    mat perturb = mvnrnd(vec(3, arma::fill::zeros), sys_var, ensemble.n_cols);
    for(int i=0; i<ensemble.n_cols; i++){
        double x_old = sol.col(i)[0];
        double y_old = sol.col(i)[1];
        double z_old = sol.col(i)[2];
        double dt_ = dt;

        double x_new = sigma * (y_old - x_old) * dt_ + x_old;
        double y_new = ( x_old * (rho - z_old) - y_old ) * dt_ + y_old;
        double z_new = ( x_old * y_old - beta * z_old ) * dt_ + z_old;

        sol.col(i)[0] = x_new;
        sol.col(i)[1] = y_new;
        sol.col(i)[2] = z_new;
    }
    return sol+perturb;
}

void lorenz63EnKF(){
    // 参数
    double ob_var = 0.1;
    double sys_var = 0.01;
    double init_var_ = 10;
    int select_every = 10;
    // 系统误差
    auto sys_error_ptr = std::make_shared<mat>(3, 3, arma::fill::eye);
    *sys_error_ptr *= sys_var;
    // 参考解
    mat ref = generate_lorenz63(10, 28, 8.0/3, 0.001, 200, vec{2,2,2}, *sys_error_ptr*0);
    mat all_ob = H_ob(ref);
    // 初始值
    vec init_ave{0., 0., 0.};
    mat init_var(3, 3, arma::fill::eye);
    init_var *= init_var_;
    // ob
    ObserveOperator ob_op = H_ob;
    auto error_ptr = std::make_shared<mat>(2, 2, arma::fill::eye);
    *error_ptr *= ob_var;

    std::vector<vec> ob_list;
    Errors ob_errors;

    for(int i=0; i<all_ob.n_cols; i++){
        // std::cout<<"in for\n";
        ob_errors.add(error_ptr);
        if(i%select_every == 0)
            ob_list.push_back(all_ob.col(i)+mvnrnd(vec(2,arma::fill::zeros), *error_ptr));
        else
            ob_list.push_back(vec());
    }
    //std::cout<<"ob-list okay\n";
    // 迭代次数
    int num_iter = ob_list.size();
    
    Errors sys_errors;
    for(int i=0; i<num_iter+1; i++)
        sys_errors.add(sys_error_ptr);
    
    int ensemble_size = 20;
    std::vector<vec> analysis_ = StochasticENKF(ensemble_size, init_ave, init_var, ob_list, num_iter, ob_errors, ob_op, model, sys_errors);
    arma::mat analysis(analysis_.size(), analysis_[0].n_rows);
    
    std::cout<<"ENKF okay\n";
    for(int i=0; i<analysis.n_rows; i++){
        analysis(i, 0) = analysis_[i](0);
        analysis(i, 1) = analysis_[i](1);
        analysis(i, 2) = analysis_[i](2);
    }
    analysis.save("analysis.csv", arma::raw_ascii);
}

int main(int argc, char** argv){
    lorenz63EnKF();
}