#include "StochasticENKF.hpp"
#include "ParticleFilter.hpp"
#include <utility>
#include <exception>
#include <boost/program_options.hpp>

using namespace arma;

extern mat compute_bino(drowvec orders, int n);

namespace config{
    drowvec derivative_orders{1.1, 1.1, 1.1};
    int window_length = 10;
    mat bino = compute_bino(derivative_orders, window_length);

    double sigma = 10., rho = 28., beta = 8./3, dt = 0.005, max_time;

    double ob_var = 1;
    double sys_var = 0.01, real_sys_var = 0.;
    double init_var_ = 10;
    int select_every = 10;

    int ensemble_size = 20;

    void (*sys_var_ptr)(mat&, mat&);

    void addRealTime(mat& ensemble, mat& sys_var){
        double var = sys_var(0, 0);

        mat real_var = mat(3, 3, arma::fill::eye);
        real_var *= var;
        
        mat perturb = arma::mvnrnd(vec{0,0,0}, real_var, ensemble.n_cols);
        ensemble.submat(0,0,2,ensemble.n_cols-1) += perturb;
    };

    void addAllTime(mat& ensemble, mat& sys_var){
        double var = sys_var(0, 0);
        if(abs(var) < 1e-12)
            return;

        mat real_var = mat(ensemble.n_rows, ensemble.n_rows, arma::fill::eye);
        real_var *= var;
        
        mat perturb = arma::mvnrnd(vec(ensemble.n_rows,arma::fill::zeros), real_var, ensemble.n_cols);
        ensemble += perturb;
    }

    void changeNoise(mat& ensemble, mat& sys_var){

    }
}

mat FracLorenz63Model(const mat& ensemble, int idx, mat sys_var){
    // 从上到下时间从近到远
    int dim = 3;
    int window = ensemble.n_rows / dim;

    // 首先计算分数阶导数
    mat fracDirivative(dim, ensemble.n_cols, arma::fill::zeros);
    for(int i=0; i<window; i++){
        for(int idx=0; idx<dim; idx++){
            fracDirivative.row(idx) += ensemble.row(i*dim + idx) * config::bino(i+1, idx);
        }
    }

    // 计算右端项
    mat rhs(dim, ensemble.n_cols, arma::fill::none);
    rhs.row(0) = config::sigma * (ensemble.row(1) - ensemble.row(0)); 
    rhs.row(1) = ensemble.row(0) % (config::rho - ensemble.row(2)) - ensemble.row(1);
    rhs.row(2) = ensemble.row(0) % ensemble.row(1) - config::beta * ensemble.row(2);

    for(int i=0; i<dim; i++){
        rhs.row(i) *= pow(config::dt, config::derivative_orders[i]);
    }

    if(window < config::window_length){
        mat ret(dim*(window+1), ensemble.n_cols, arma::fill::none);
        ret.submat(0,0,dim-1,ensemble.n_cols-1) = rhs - fracDirivative;
        ret.submat(dim,0,ret.n_rows-1,ret.n_cols-1) = ensemble;
        if(config::sys_var_ptr)
            config::sys_var_ptr(ret, sys_var);
        return ret;
    }else{
        mat ret(ensemble.n_rows, ensemble.n_cols, arma::fill::none);
        ret.submat(0,0,dim-1,ensemble.n_cols-1) = rhs - fracDirivative;
        ret.submat(dim,0,ret.n_rows-1,ret.n_cols-1) = ensemble.submat(0,0,ensemble.n_rows-dim-1,ensemble.n_cols-1);
        if(config::sys_var_ptr)
            config::sys_var_ptr(ret, sys_var);
        return ret;
    }
}

mat generateFracLorenz63(double dt, double max_time, vec v0, mat sys_var){
    int iter_num = max_time / dt;
    if(iter_num * dt < max_time)
        iter_num++;

    mat result(iter_num+1, 3, arma::fill::none);
    result.row(0) = v0.t();

    // 保存config::dt和window
    double pre_dt = config::dt;
    config::dt = dt;
    int pre_window = config::window_length;
    mat pre_bino = config::bino;
    int window = 20;
    std::cin>>window;
    config::window_length = window;
    config::bino = compute_bino(config::derivative_orders, config::window_length);

    std::cout<<"using window length "<<config::window_length<<" to generate reference solution\n";
    for(int i=1; i<iter_num+1; i++){
        if(i * dt <= max_time){
            // 这里是正常的dt
            v0 = FracLorenz63Model(v0, i, sys_var);
            result.row(i) = v0.subvec(0, 2).t();
        }else{
            config::dt = max_time - (i - 1) * dt;
            v0 = FracLorenz63Model(v0, i, sys_var);
            result.row(i) = v0.subvec(0, 2).t();
        }
    }

    // 恢复config::dt和Window
    config::dt = pre_dt;
    config::window_length = pre_window;
    config::bino = pre_bino;

    return result;
}

mat compute_bino(drowvec orders, int n){
    mat bino(n+1, orders.n_cols, arma::fill::none);
    
    bino.row(0) = drowvec(orders.n_cols, arma::fill::ones);
    // std::cout<<"compute okay\n";
    for(int i=1; i<n+1; i++){
        bino.row(i) = (1. - (1. + orders) / i ) % bino.row(i-1);
    }

    return bino;
}

void test_frac_lorenz63(){
    mat sol = generateFracLorenz63(0.005, 100, {0.1,0.1,0.1}, mat(3,3,arma::fill::zeros));
    // config::bino = compute_bino(config::derivative_orders, config::window_length);
    sol.save("./data/lorenz63.csv", arma::raw_ascii);
}

void fracLorenz63EnKF(){
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
    mat ref = generateFracLorenz63(config::dt, config::max_time, v0, *sys_error_ptr*config::real_sys_var);
    ref.save("./data/lorenz63.csv", arma::raw_ascii);
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
    mat all_ob = H_ob(ref.t());
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
    Errors ob_errors;

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
    
    Errors sys_errors;
    for(int i=0; i<num_iter+1; i++)
        sys_errors.add(sys_error_ptr);
    
    int ensemble_size = config::ensemble_size;

    std::cout<<"ENKF ready\n";
    auto ENKFResult = StochasticENKF(ensemble_size, init_ave, init_var, ob_list, num_iter, ob_errors, ob_op, FracLorenz63Model, sys_errors);
    std::vector<vec> analysis_ = std::get<0>(ENKFResult);
    std::vector<double> skewness_ = std::get<1>(ENKFResult);
    std::vector<double> kurtosis_ = std::get<2>(ENKFResult);
    std::cout<<"ENKF okay\n";

    arma::mat analysis(analysis_.size(), 3);
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

void fracLorenz63EnKF_version2(){
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
    mat ref = generateFracLorenz63(config::dt, config::max_time, v0, *sys_error_ptr*config::real_sys_var);
    ref.save("./data/lorenz63.csv", arma::raw_ascii);
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
    mat all_ob = H_ob(ref.t());
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
    Errors ob_errors;

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
    
    Errors sys_errors;
    for(int i=0; i<num_iter+1; i++)
        sys_errors.add(sys_error_ptr);
    
    int ensemble_size = config::ensemble_size;

    std::cout<<"ENKF ready\n";
    config::window_length = 20000;
    config::bino = compute_bino(config::derivative_orders, config::window_length);
    auto ENKFResult = FStochasticENKF<3, 5>(ensemble_size, init_ave, init_var, ob_list, 
        num_iter, ob_errors, ob_op, FracLorenz63Model, sys_errors, 0.1*arma::eye(15, 15));
    std::vector<vec> analysis_ = ENKFResult;
    // std::vector<double> skewness_ = std::get<1>(ENKFResult);
    // std::vector<double> kurtosis_ = std::get<2>(ENKFResult);
    std::cout<<"ENKF okay\n";

    arma::mat analysis(analysis_.size(), 3);
    for(int i=0; i<analysis.n_rows; i++){
        analysis(i, 0) = analysis_[i](0);
        analysis(i, 1) = analysis_[i](1);
        analysis(i, 2) = analysis_[i](2);
    }
    // arma::mat skewness(skewness_.size(), 1);
    // arma::mat kurtosis(skewness_.size(), 1);
    // for(int i=0; i<skewness.n_rows; i++){
    //     skewness(i, 0) = skewness_[i];
    //     kurtosis(i, 0) = kurtosis_[i];
    // }

    analysis.save("./data/analysis.csv", arma::raw_ascii);
    // skewness.save("./data/skewness.csv", arma::raw_ascii);
    // kurtosis.save("./data/kurtosis.csv", arma::raw_ascii);
}

int main(int argc, char** argv){
    using namespace boost::program_options;
    using namespace config;

    int version = 1;
    std::string problem, sys_var_type;

    options_description cmd("Fractional lorenz63 EnKF");
    cmd.add_options()("problem,p", value<std::string>(&problem)->default_value("ENKF"), "type: ENKF or Particle");
    cmd.add_options()("version,e", value<int>(&version)->default_value(1), "ENKF version");
    cmd.add_options()("sigma,s", value<double>(&sigma)->default_value(10), "sigma");
    cmd.add_options()("rho,r", value<double>(&rho)->default_value(28), "rho");
    cmd.add_options()("beta,b", value<double>(&beta)->default_value(8.0/3), "beta");
    cmd.add_options()("window,w", value<int>(&window_length)->default_value(10), "window length");
    cmd.add_options()("ob_var,o", value<double>(&ob_var)->default_value(0.1), "ob_error");
    cmd.add_options()("sys_var,v", value<double>(&sys_var)->default_value(0.1), "system_error");
    cmd.add_options()("sys_var_type,y", value<std::string>(&sys_var_type)->default_value("real"), "system_error_type");
    cmd.add_options()("init_var,i", value<double>(&init_var_)->default_value(10), "init_error");
    cmd.add_options()("real_sys_var,a", value<double>(&real_sys_var)->default_value(1), "real_system_error");
    cmd.add_options()("select,l", value<int>(&select_every)->default_value(10), "select every");
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

    bino = compute_bino(derivative_orders, window_length);

    if(sys_var_type == "real")
        sys_var_ptr = addRealTime;
    else if(sys_var_type == "all")
        sys_var_ptr = addAllTime;
    else if(sys_var_type == "noise")
        sys_var_ptr = changeNoise;
    else
        sys_var_ptr = nullptr;
    
    arma::arma_rng::set_seed_random();
    // lorenz63EnKF();
    if(problem == "ENKF")
        if(version == 1)
            fracLorenz63EnKF();
        else
            fracLorenz63EnKF_version2();
    else
        throw("Not implemented yet");

    std::cout<<"problem type: "<<problem<<'\n'
            <<"sigma: "<<sigma<<'\n'
            <<"rho: "<<rho<<'\n'
            <<"beta: "<<beta<<'\n'
            <<"init_var: "<<init_var_<<'\n'
            <<"ob_var: "<<ob_var<<'\n'
            <<"sys_var: "<<sys_var<<'\n'
            <<"sys_var_type: "<<sys_var_type<<'\n'
            <<"real_sys_var: "<<real_sys_var<<'\n'
            <<"select every: "<<select_every<<'\n'
            <<"ensemble size: "<<ensemble_size<<'\n'
            <<"max time: "<<max_time<<'\n'
            <<"ENKF version: "<<version<<'\n';
}