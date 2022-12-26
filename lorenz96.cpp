#include "StochasticENKF.hpp"

#include <iostream>
#include <boost/program_options.hpp>

using namespace arma;

namespace config{
    int dim = 40;
    int ob_dim = 6;
    double F = 8;
    double dt = 0.005;
    double t_max = 20;

    double ob_var = 0.01;
    double sys_var = 0.01, real_sys_var = 0.;
    double init_var_ = 10;
    int select_every = 10;

    int ensemble_size = 20;
}

mat lorenz96model(const mat& ensemble, int idx, const mat& sys_var){
    int M = ensemble.n_rows, N = ensemble.n_cols;
    mat newer(M, N, arma::fill::none);

    // forward 
    // for(int j=0; j<N; j++){
    //     for(int i=0; i<M; i++){
    //         int pre = (i + M - 1) % M;
    //         int far = (i + M - 2) % M;
    //         int next = (i + 1) % M;

    //         double derivative = ensemble(pre, j) * (ensemble(next, j) - ensemble(far, j)) - ensemble(i, j) + config::F;
    //         double value = derivative * config::dt + ensemble(i, j);
    //         newer(i, j) = value;
    //     }
    // }

    for(int i=0; i<M; i++){
        int pre = (i + M - 1) % M;
        int far = (i + M - 2) % M;
        int next = (i + 1) % M;

        mat derivative = ensemble.row(pre) % (ensemble.row(next) - ensemble.row(far)) - ensemble.row(i) + config::F;
        mat value = derivative * config::dt + ensemble.row(i);
        newer.row(i) = value;
    }

    // add perturbation
    mat temp, perturb(M, N, arma::fill::none);
    if(arma::inv(temp, sys_var)){
        perturb = arma::mvnrnd(vec(M,arma::fill::zeros), sys_var, N);
        newer += perturb;
    }

    return newer;
}


// we'll realise the observation operator as random matrix
// so here we omit it

mat generate_lorenz96(vec v0, double F, double time_max, double dt, const mat& sys_var){
    int num_iter = time_max / dt;
    if(num_iter * dt < time_max)
        num_iter++;
    int dim = v0.n_rows;
    
    // initialize result matrix 
    mat result(num_iter+1, dim, arma::fill::none);
    for(int i=0; i<dim; i++){
        result(0, i) = v0(i);
    }

    // save config::dt for future recovery
    double former_dt = config::dt;
    config::dt = dt;
    for(int i=0; i<num_iter; i++){
        v0 = lorenz96model(v0, i, sys_var);
        // save new simulation
        for(int j=0; j<dim; j++){
            result(i+1, j) = v0(j);
        }
    }

    // recovery config::dt
    config::dt = former_dt;

    return result;
}

void test_lorenz96(){
    vec v0 = arma::randn(40);
    mat sys_var(40, 40, arma::fill::zeros);
    mat sol = generate_lorenz96(v0, 8, 100, 0.01, sys_var);
    sol.save("./data/lorenz96.csv", arma::raw_ascii);
}

void lorenz96EnKF(){
    // 参数
    int dim = config::dim;
    int ob_dim = config::ob_dim;
    double ob_var = config::ob_var;
    double sys_var = config::sys_var;
    double init_var_ = config::init_var_;
    int select_every = config::select_every;
    // ob算子
    mat H = arma::randn(ob_dim, dim);
    auto H_ob = [&H](const mat& ensemble){
        return H * ensemble;
    };
    // 系统误差
    auto sys_error_ptr = std::make_shared<mat>(dim, dim, arma::fill::eye);
    *sys_error_ptr *= sys_var;
    // 参考解
    vec v0 = arma::randn(dim);
    mat ref = generate_lorenz96(v0, config::F, config::t_max, config::dt, *sys_error_ptr*config::real_sys_var);
    mat all_ob = H_ob(ref.t());
    ref.save("./data/lorenz96.csv", arma::raw_ascii);
    // 初始值
    vec init_ave(dim, arma::fill::zeros);
    // init_ave = v0;
    mat init_var(dim, dim, arma::fill::eye);
    init_var *= init_var_;
    // ob
    auto ob_op = H_ob;
    auto error_ptr = std::make_shared<mat>(ob_dim, ob_dim, arma::fill::eye);
    *error_ptr *= ob_var;

    std::vector<vec> ob_list;
    Errors ob_errors;

    for(int i=0; i<all_ob.n_cols; i++){
        // std::cout<<"in for\n";
        ob_errors.add(error_ptr);
        if(i%select_every == 0)
            ob_list.push_back(all_ob.col(i)+mvnrnd(vec(ob_dim, arma::fill::zeros), *error_ptr));
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
    auto ENKFResult = StochasticENKF(ensemble_size, init_ave, init_var, ob_list, num_iter, ob_errors, ob_op, lorenz96model, sys_errors);
    std::vector<vec> analysis_ = std::get<0>(ENKFResult);
    std::vector<double> skewness_ = std::get<1>(ENKFResult);
    std::vector<double> kurtosis_ = std::get<2>(ENKFResult);
    std::cout<<"ENKF okay\n";

    arma::mat analysis(analysis_.size(), analysis_[0].n_rows);
    for(int i=0; i<analysis.n_rows; i++){
        for(int j=0; j<dim; j++){
            analysis(i, j) = analysis_[i](j);
        }
    }
    arma::mat skewness(skewness_.size(), 1);
    arma::mat kurtosis(skewness_.size(), 1);
    for(int i=0; i<skewness.n_rows; i++){
        skewness(i, 0) = skewness_[i];
        kurtosis(i, 0) = kurtosis_[i];
    }

    analysis.save("./data/analysis.csv", arma::raw_ascii);
    skewness.save("./data/skewness.csv", arma::raw_ascii);
    kurtosis.save("./data/kurtosis.csv", arma::raw_ascii);
}

int main(int argc, char** argv){
    using namespace boost::program_options;
    using namespace config;

    options_description cmd("lorenz96 EnKF");
    cmd.add_options()("dim,d", value<int>(&dim)->default_value(40), "dimension of system");
    cmd.add_options()("ob_dim,o", value<int>(&ob_dim)->default_value(6), "ob dimension");
    cmd.add_options()("F,F", value<double>(&F)->default_value(8.0), "F");
    cmd.add_options()("ob_var,b", value<double>(&ob_var)->default_value(0.1), "ob_error");
    cmd.add_options()("sys_var,v", value<double>(&sys_var)->default_value(0.01), "system_error");
    cmd.add_options()("init_var,i", value<double>(&init_var_)->default_value(10), "init_error");
    cmd.add_options()("real_sys_var,r", value<double>(&real_sys_var)->default_value(0.), "real_system_error");
    cmd.add_options()("select,s", value<int>(&select_every)->default_value(10), "select every");
    cmd.add_options()("size,n", value<int>(&ensemble_size)->default_value(20), "ensemble size");
    cmd.add_options()("max_time,t", value<double>(&t_max)->default_value(20), "max time");

    variables_map map;
    store(parse_command_line(argc, argv, cmd), map);
    notify(map);

    lorenz96EnKF();

    std::cout<<"dim: "<<dim<<'\n'
            <<"ob_dim: "<<ob_dim<<'\n'
            <<"F: "<<F<<'\n'
            <<"init_var: "<<init_var_<<'\n'
            <<"ob_var: "<<ob_var<<'\n'
            <<"sys_var: "<<sys_var<<'\n'
            <<"real_sys_var: "<<real_sys_var<<'\n'
            <<"select every: "<<select_every<<'\n'
            <<"ensemble size: "<<ensemble_size<<'\n'
            <<"max time: "<<t_max<<'\n';
}