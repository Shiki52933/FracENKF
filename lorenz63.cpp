#include "StochasticENKF.hpp"
#include <armadillo>
#include <iostream>
#include <boost/program_options.hpp>


mat generate_lorenz63(double sigma, double rho, double beta, double dt, double t_max, vec x0, mat sys_var){
    int num_iter = t_max / dt;
    if( dt * num_iter < t_max )
        num_iter++;

    mat sol(3, num_iter+1);

    mat temp, perturb(3, num_iter, arma::fill::zeros);
    if(arma::inv(temp, sys_var))
        perturb = mvnrnd(vec(3,arma::fill::zeros), sys_var, num_iter);

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

namespace config{
    double sigma = 10, rho = 28, beta = 8.0/3;
    double dt = 0.001;

    double ob_var = 1;
    double sys_var = 0.01, real_sys_var = 0.;
    double init_var_ = 10;
    int select_every = 10;

    int ensemble_size = 20;
}

mat model(mat& ensemble, int idx, mat& sys_var){
    double sigma = config::sigma, rho = config::rho, beta = config::beta;
    double dt = config::dt;

    mat sol = ensemble;

    mat temp, perturb(3, ensemble.n_cols, arma::fill::zeros);
    if(arma::inv(temp, sys_var))
        perturb = mvnrnd(vec(3, arma::fill::zeros), sys_var, ensemble.n_cols);

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
    double ob_var = config::ob_var;
    double sys_var = config::sys_var;
    double init_var_ = config::init_var_;
    int select_every = config::select_every;
    // 系统误差
    auto sys_error_ptr = std::make_shared<mat>(3, 3, arma::fill::eye);
    *sys_error_ptr *= sys_var;
    // 参考解
    mat ref = generate_lorenz63(config::sigma, config::rho, config::beta, config::dt, 200, vec{2,2,2}, *sys_error_ptr*config::real_sys_var);
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
    
    int ensemble_size = config::ensemble_size;
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
    using namespace boost::program_options;
    using namespace config;

    options_description cmd("lorenz63 EnKF");
    cmd.add_options()("sigma,s", value<double>(&sigma)->default_value(10), "sigma");
    cmd.add_options()("rho,r", value<double>(&rho)->default_value(28), "rho");
    cmd.add_options()("beta,b", value<double>(&beta)->default_value(8.0/3), "beta");
    cmd.add_options()("ob_var,o", value<double>(&ob_var)->default_value(0.1), "ob_error");
    cmd.add_options()("sys_var,v", value<double>(&sys_var)->default_value(0.01), "system_error");
    cmd.add_options()("init_var,i", value<double>(&init_var_)->default_value(10), "init_error");
    cmd.add_options()("real_sys_var,rs", value<double>(&real_sys_var)->default_value(0.), "real_system_error");
    cmd.add_options()("select,sl", value<int>(&select_every)->default_value(10), "select every");
    cmd.add_options()("size,n", value<int>(&ensemble_size)->default_value(20), "ensemble size");
    
    variables_map map;
    store(parse_command_line(argc, argv, cmd), map);
    notify(map);

/*
    config::sigma = map["sigma"].as<double>();
    config::rho = map["rho"].as<double>();
    config::beta = map["beta"].as<double>();
    config::ob_var = map["ob error"].as<double>();
    config::sys_var = map["system error"].as<double>();
    config::real_sys_var = map["real system error"].as<double>();
    config::select_every = map["select every"].as<int>();
    config::ensemble_size = map["ensemble size"].as<int>();
*/ 

    lorenz63EnKF();

    std::cout<<"sigma: "<<sigma<<'\n'
            <<"rho: "<<rho<<'\n'
            <<"beta: "<<beta<<'\n'
            <<"ob_var: "<<ob_var<<'\n'
            <<"sys_var: "<<sys_var<<'\n'
            <<"real_sys_var: "<<real_sys_var<<'\n'
            <<"select every: "<<select_every<<'\n'
            <<"ensemble size: "<<ensemble_size<<'\n';
}