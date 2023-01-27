#include "StochasticENKF.hpp"
#include "ParticleFilter.hpp"
#include "3DVar.hpp"
#include "fUKF.hpp"
#include "fEKF.hpp"
#include <armadillo>
#include <iostream>
#include <boost/program_options.hpp>

using namespace arma;
using namespace shiki;

mat generate_Lorenz63(double sigma, double rho, double beta, double dt, double t_max, vec x0, mat sys_var){
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


namespace config{
    double sigma = 10, rho = 28, beta = 8.0/3;
    double dt = 0.005, max_time=5;

    double ob_var = 1;
    double sys_var = 0.01, real_sys_var = 0.;
    double init_var_ = 10;
    int select_every = 10;

    int ensemble_size = 20;
}

mat Lorenz63_rhs(const mat& ensemble){
    // 计算右端项
    const int dim = 3;
    mat rhs(dim, ensemble.n_cols, arma::fill::none);
    rhs.row(0) = config::sigma * (ensemble.row(1) - ensemble.row(0)); 
    rhs.row(1) = ensemble.row(0) % (config::rho - ensemble.row(2)) - ensemble.row(1);
    rhs.row(2) = ensemble.row(0) % ensemble.row(1) - config::beta * ensemble.row(2);
    return rhs;
}

mat Lorenz63_linearize(vec mean){
    return mat{
        {-config::sigma, config::sigma, 0},
        {config::rho - mean(2), -1, -mean(0)},
        {mean(1), mean(0), -config::beta}
    };
}

mat model(const mat& ensemble, int idx, mat sys_var){
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

void Lorenz63_EnKF(){
    // 参数
    double ob_var = config::ob_var;
    double sys_var = config::sys_var;
    double init_var_ = config::init_var_;
    int select_every = config::select_every;
    // 系统误差
    auto sys_error_ptr = std::make_shared<mat>(3, 3, arma::fill::eye);
    *sys_error_ptr *= sys_var;
    // 参考解
    mat ref = generate_Lorenz63(config::sigma, config::rho, config::beta, config::dt, config::max_time, vec{2,2,2}, *sys_error_ptr*config::real_sys_var);
    
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
    errors ob_errors;

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
    
    errors sys_errors;
    for(int i=0; i<num_iter+1; i++)
        sys_errors.add(sys_error_ptr);
    
    int ensemble_size = config::ensemble_size;
    auto ENKFResult = stochastic_ENKF_normal_test(
        ensemble_size, num_iter, 
        init_ave, init_var, 
        ob_list, ob_op, ob_errors, 
        model, sys_errors
        );
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

void Lorenz63_3DVar(){
    // 参数
    double ob_var = config::ob_var;
    double sys_var = config::sys_var;
    double init_var_ = config::init_var_;
    int select_every = config::select_every;
    // 系统误差
    auto sys_error_ptr = std::make_shared<mat>(3, 3, arma::fill::eye);
    *sys_error_ptr *= sys_var;
    // 参考解
    mat ref = generate_Lorenz63(config::sigma, config::rho, config::beta, config::dt, config::max_time, vec{2,2,2}, *sys_error_ptr*config::real_sys_var);
    
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
    errors ob_errors;

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
    
    errors sys_errors;
    for(int i=0; i<num_iter+1; i++)
        sys_errors.add(sys_error_ptr);
    
    int ensemble_size = config::ensemble_size;
    auto ThreeDVarResult = var_3d(
        3, 1, 
        init_ave, *sys_error_ptr,
        ob_list, H, ob_errors, 
        model, sys_errors
        );
    std::vector<vec>& analysis_ = ThreeDVarResult;
    std::cout<<"ENKF okay\n";

    arma::mat analysis(analysis_.size(), analysis_[0].n_rows);
    for(int i=0; i<analysis.n_rows; i++){
        analysis(i, 0) = analysis_[i](0);
        analysis(i, 1) = analysis_[i](1);
        analysis(i, 2) = analysis_[i](2);
    }

    analysis.save("./data/analysis.csv", arma::raw_ascii);
}

void Lorenz63_particle(){
    // 参数
    double ob_var = config::ob_var;
    double sys_var = config::sys_var;
    double init_var_ = config::init_var_;
    int select_every = config::select_every;

    // 系统误差
    auto sys_error_ptr = std::make_shared<mat>(3, 3, arma::fill::eye);
    *sys_error_ptr *= sys_var;

    // 参考解
    mat ref = generate_Lorenz63(config::sigma, config::rho, config::beta, config::dt, config::max_time, vec{2,2,2}, *sys_error_ptr*config::real_sys_var);
    
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
    errors ob_errors;
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
    errors sys_errors;
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
    auto particle = particle_filter(ensemble, model, likehood, ob_list, sys_errors);
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

void Lorenz63_UKf(){
    // 参数
    double ob_var = config::ob_var;
    double sys_var = config::sys_var;
    double init_var_ = config::init_var_;
    int select_every = config::select_every;
    // 系统误差
    auto sys_error_ptr = std::make_shared<mat>(3, 3, arma::fill::eye);
    *sys_error_ptr *= sys_var;
    // 参考解
    vec v0 = randn(3);
    mat ref = generate_Lorenz63(
        config::sigma, config::rho, config::beta,
        config::dt, config::max_time, v0, *sys_error_ptr
        );
    std::cout<<"reference solution okay\n";

    mat H = arma::randn(2, 3);
    auto H_ob = [&H](const mat& ensemble) -> mat {
        // std::cout<<"ensemble n_rows: "<<ensemble.n_rows<<"\tn_cols: "<<ensemble.n_cols<<'\n';
        if(ensemble.n_rows == 3){
            return H * ensemble;
        }else{
            // std::cout<<"start multiplication\n";
            mat real_time = ensemble.submat(0, 0, 2, ensemble.n_cols-1);
            mat ret = H * real_time;
            // std::cout<<"end multiplication\n";
            return ret;
        }
    };
    // mat temp = ref.t();
    mat all_ob = H_ob(ref);
    std::cout<<"all ob okay\n";
    // 初始值
    vec init_ave = v0;
    mat init_var(3, 3, arma::fill::eye);
    init_var *= init_var_;
    // ob
    auto ob_op = H_ob;
    auto error_ptr = std::make_shared<mat>(2, 2, arma::fill::eye);
    *error_ptr *= ob_var;

    std::vector<vec> ob_list;
    errors ob_errors;

    for(int i=0; i<all_ob.n_cols; i++){
        // std::cout<<"in for\n";
        ob_errors.add(error_ptr);
        if(i%select_every == 0)
            ob_list.push_back(all_ob.col(i)+mvnrnd(vec(2,arma::fill::zeros), *error_ptr));
        else
            ob_list.push_back(vec());
    }
    std::cout<<"ob-list okay\n";
    // 迭代次数
    int num_iter = ob_list.size();
    
    errors sys_errors;
    for(int i=0; i<num_iter+1; i++)
        sys_errors.add(sys_error_ptr);
    
    std::cout<<"UKF ready\n";
    // config::bino = compute_bino(config::derivative_orders, config::window_length);
    auto ENKFResult = fUKF(
        0.5, 2, 0,
        3, drowvec{1.,1.,1.}, config::dt,
        init_ave, init_var,
        ob_list, H, ob_errors,
        Lorenz63_rhs, sys_errors, 0.1*arma::eye(3,3));
    mat& analysis = ENKFResult;
    std::cout<<"UKF okay\n";

    analysis.save("./data/analysis.csv", arma::raw_ascii);
}

void Lorenz63_EKf(){
    // 参数
    double ob_var = config::ob_var;
    double sys_var = config::sys_var;
    double init_var_ = config::init_var_;
    int select_every = config::select_every;
    // 系统误差
    auto sys_error_ptr = std::make_shared<mat>(3, 3, arma::fill::eye);
    *sys_error_ptr *= sys_var;
    // 参考解
    vec v0 = randn(3);
    mat ref = generate_Lorenz63(
        config::sigma, config::rho, config::beta,
        config::dt, config::max_time, v0, *sys_error_ptr
        );
    std::cout<<"reference solution okay\n";

    mat H = arma::randn(2, 3);
    auto H_ob = [&H](const mat& ensemble) -> mat {
        // std::cout<<"ensemble n_rows: "<<ensemble.n_rows<<"\tn_cols: "<<ensemble.n_cols<<'\n';
        if(ensemble.n_rows == 3){
            return H * ensemble;
        }else{
            // std::cout<<"start multiplication\n";
            mat real_time = ensemble.submat(0, 0, 2, ensemble.n_cols-1);
            mat ret = H * real_time;
            // std::cout<<"end multiplication\n";
            return ret;
        }
    };
    // mat temp = ref.t();
    mat all_ob = H_ob(ref);
    std::cout<<"all ob okay\n";
    // 初始值
    vec init_ave{0., 0., 0.};
    mat init_var(3, 3, arma::fill::eye);
    init_var *= init_var_;
    // ob
    auto ob_op = H_ob;
    auto error_ptr = std::make_shared<mat>(2, 2, arma::fill::eye);
    *error_ptr *= ob_var;

    std::vector<vec> ob_list;
    errors ob_errors;

    for(int i=0; i<all_ob.n_cols; i++){
        // std::cout<<"in for\n";
        ob_errors.add(error_ptr);
        if(i%select_every == 0)
            ob_list.push_back(all_ob.col(i)+mvnrnd(vec(2,arma::fill::zeros), *error_ptr));
        else
            ob_list.push_back(vec());
    }
    std::cout<<"ob-list okay\n";
    // 迭代次数
    int num_iter = ob_list.size();
    
    errors sys_errors;
    for(int i=0; i<num_iter+1; i++)
        sys_errors.add(sys_error_ptr);
    
    std::cout<<"EKF ready\n";
    auto ENKFResult = fEKF(
        3, drowvec{1.,1.,1.}, config::dt, 0.1*arma::eye(3,3),
        init_ave, init_var,
        ob_list, H, ob_errors,
        model, Lorenz63_linearize, sys_errors);
    mat& analysis = ENKFResult;
    std::cout<<"EKF okay\n";

    analysis.save("./data/analysis.csv", arma::raw_ascii);
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

    arma::arma_rng::set_seed_random();
    // lorenz63EnKF();
    if(problem == "ENKF")
        Lorenz63_EnKF();
    else if(problem == "particle")
        Lorenz63_particle();
    else if(problem == "3d-var")
        Lorenz63_3DVar();
    else if(problem == "UKF")
        Lorenz63_UKf();
    else if(problem == "EKF")
        Lorenz63_EKf();
    else
        throw std::runtime_error("not supported filter algorithm");

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