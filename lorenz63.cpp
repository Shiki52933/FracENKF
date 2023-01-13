#include "StochasticENKF.hpp"
#include "ParticleFilter.hpp"
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
        perturb = arma::mvnrnd(vec(3,arma::fill::zeros), sys_var, num_iter);

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
    sol_t.save("./data/lorenz63.csv", arma::raw_ascii);
    return sol;
}

// mat H_ob(mat& ensemble){
//     vec a{0, 1, 1};
//     vec b{1, 1, 0};
//     mat H(3, 2);
//     H.col(0) = a;
//     H.col(1) = b;
//     H = H.t();
//     return H * ensemble;
//     /*
//     mat ob(2, ensemble.n_cols);
//     for(int i=0; i<ensemble.n_cols; i++){
//         ob(0, i) = sin(ensemble(0, i)) + cos(ensemble(1, i));
//         ob(1, i) = sin(ensemble(1, i)) + cos(ensemble(2, i));
//     }
//     return ob;
//     */
// }

namespace config{
    double sigma = 10, rho = 28, beta = 8.0/3;
    double dt = 0.005, max_time=5;

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
        perturb = arma::mvnrnd(vec(3, arma::fill::zeros), sys_var, ensemble.n_cols);

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
    mat ref = generate_lorenz63(config::sigma, config::rho, config::beta, config::dt, config::max_time, vec{2,2,2}, *sys_error_ptr*config::real_sys_var);
    
    mat H = arma::randn(2, 3);
    auto H_ob = [&H](const mat& ensemble){
        return H * ensemble;
    };
    mat all_ob = H_ob(ref);
    // 初始值
    vec init_ave{0., 0., 0.};
    mat init_var(3, 3, arma::fill::eye);
    init_var *= init_var_;
    // ob
    auto ob_op = H_ob;
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
    auto ENKFResult = StochasticENKF(ensemble_size, init_ave, init_var, ob_list, num_iter, ob_errors, ob_op, model, sys_errors);
    std::vector<vec> analysis_ = std::get<0>(ENKFResult);
    std::vector<double> skewness_ = std::get<1>(ENKFResult);
    std::vector<double> kurtosis_ = std::get<2>(ENKFResult);
    std::cout<<"ENKF okay\n";

    arma::mat analysis(analysis_.size(), analysis_[0].n_rows);
    for(int i=0; i<analysis.n_rows; i++){
        analysis(i, 0) = analysis_[i](0);
        analysis(i, 1) = analysis_[i](1);
        analysis(i, 2) = analysis_[i](2);
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

void lorenz63particle(){
    // 参数
    double ob_var = config::ob_var;
    double sys_var = config::sys_var;
    double init_var_ = config::init_var_;
    int select_every = config::select_every;

    // 系统误差
    auto sys_error_ptr = std::make_shared<mat>(3, 3, arma::fill::eye);
    *sys_error_ptr *= sys_var;

    // 参考解
    mat ref = generate_lorenz63(config::sigma, config::rho, config::beta, config::dt, config::max_time, vec{2,2,2}, *sys_error_ptr*config::real_sys_var);
    
    // ob算子
    mat H = arma::randn(2, 3);
    auto error_ptr = std::make_shared<mat>(2, 2, arma::fill::eye);
    *error_ptr *= ob_var;

    // 辅助lambda表达式
    auto H_ob = [&H](const mat& ensemble){
        return H * ensemble;
    };
    auto likehood = [&H, &error_ptr](vec solid, vec ob){
        vec ob_ = H * solid;
        // std::cout<<"enter likehood\n";
        vec misfit = ob_ - ob;
        mat likehood = -1./2 * misfit.t() * error_ptr->i() * misfit;
        // std::cout<<"end likehood\n";
        return exp(likehood(0, 0));
    };

    std::vector<vec> ob_list;
    Errors ob_errors;
    mat all_ob = H_ob(ref);

    for(int i=0; i<all_ob.n_cols; i++){
        // std::cout<<"in for\n";
        ob_errors.add(error_ptr);
        if(i%select_every == 0)
            ob_list.push_back(all_ob.col(i)+mvnrnd(vec(2,arma::fill::zeros), *error_ptr));
        else
            ob_list.push_back(vec());
    }

    // 系统误差
    Errors sys_errors;
    for(int i=0; i<ob_list.size()+1; i++)
        sys_errors.add(sys_error_ptr);
    
    // 初始值
    int size = config::ensemble_size;
    vec init_ave{0., 0., 0.};
    mat init_var(3, 3, arma::fill::eye);
    init_var *= init_var_;
    mat ensemble = mvnrnd(init_ave, init_var, size);

    // particle filter
    std::cout<<"ready for particle filter\n";
    auto particle = ParticleFilter(ensemble, model, likehood, ob_list, sys_errors);
    std::cout<<"particle filter ended\n";

    // 后处理
    mat assimilated(particle.size(), 3, arma::fill::none);

    for(int i=0; i<particle.size(); i++){
        vec temp(3, arma::fill::zeros);
        for(int j=0; j<particle[i].first.n_cols; j++){
            temp += particle[i].first.col(j) * particle[i].second(j);
        }
        assimilated(i, 0) = temp(0);
        assimilated(i, 1) = temp(1);
        assimilated(i, 2) = temp(2);
    }
    assimilated.save("./data/analysis.csv", arma::raw_ascii);
}

int main(int argc, char** argv){
    using namespace boost::program_options;
    using namespace config;

    std::string problem;

    options_description cmd("lorenz63 EnKF");
    cmd.add_options()("problem,p", value<std::string>(&problem)->default_value("ENKF"), "type: ENKF or Particle");
    cmd.add_options()("sigma,s", value<double>(&sigma)->default_value(10), "sigma");
    cmd.add_options()("rho,r", value<double>(&rho)->default_value(28), "rho");
    cmd.add_options()("beta,b", value<double>(&beta)->default_value(8.0/3), "beta");
    cmd.add_options()("ob_var,o", value<double>(&ob_var)->default_value(0.1), "ob_error");
    cmd.add_options()("sys_var,v", value<double>(&sys_var)->default_value(0.1), "system_error");
    cmd.add_options()("init_var,i", value<double>(&init_var_)->default_value(10), "init_error");
    cmd.add_options()("real_sys_var,y", value<double>(&real_sys_var)->default_value(1), "real_system_error");
    cmd.add_options()("select,c", value<int>(&select_every)->default_value(10), "select every");
    cmd.add_options()("size,n", value<int>(&ensemble_size)->default_value(20), "ensemble size");
    cmd.add_options()("time,t", value<double>(&max_time)->default_value(50), "max_time");
    
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
    arma::arma_rng::set_seed_random();
    // lorenz63EnKF();
    if(problem == "ENKF")
        lorenz63EnKF();
    else
        lorenz63particle();

    std::cout<<"problem type: "<<problem<<'\n'
            <<"sigma: "<<sigma<<'\n'
            <<"rho: "<<rho<<'\n'
            <<"beta: "<<beta<<'\n'
            <<"init_var: "<<init_var_<<'\n'
            <<"ob_var: "<<ob_var<<'\n'
            <<"sys_var: "<<sys_var<<'\n'
            <<"real_sys_var: "<<real_sys_var<<'\n'
            <<"select every: "<<select_every<<'\n'
            <<"ensemble size: "<<ensemble_size<<'\n'
            <<"max time: "<<max_time<<'\n';
}