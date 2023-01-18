#include "StochasticENKF.hpp"
#include "ParticleFilter.hpp"

#include <iostream>
#include <boost/program_options.hpp>

using namespace arma;

mat compute_bino(drowvec orders, int n){
    mat bino(n+1, orders.n_cols, arma::fill::none);
    
    bino.row(0) = drowvec(orders.n_cols, arma::fill::ones);
    // std::cout<<"compute okay\n";
    for(int i=1; i<n+1; i++){
        bino.row(i) = (1. - (1. + orders) / i ) % bino.row(i-1);
    }

    return bino;
}

namespace config{
    int dim = 40;
    int ob_dim = 6;
    double F = 8;
    double dt = 0.005;
    double t_max = 200;

    int window_length = 10;
    drowvec orders = randn<drowvec>(dim, distr_param(0.8, 1e-1));
    mat bino = compute_bino(orders, (int)200/dt);

    double ob_var = 0.01;
    double sys_var = 0.01, real_sys_var = 0.;
    double init_var_ = 10;
    int select_every = 10;

    int ensemble_size = 20;

    void (*sys_var_ptr)(mat&, const mat&);

    void addRealTime(mat& ensemble, const mat& sys_var){
        double var = sys_var(0, 0);

        mat real_var = mat(dim, dim, arma::fill::eye);
        real_var *= var;
        
        mat perturb = arma::mvnrnd(vec(dim,arma::fill::zeros), real_var, ensemble.n_cols);
        ensemble.submat(0,0,dim-1,ensemble.n_cols-1) += perturb;
    };

    void addAllTime(mat& ensemble, const mat& sys_var){
        double var = sys_var(0, 0);
        if(abs(var) < 1e-12)
            return;

        mat real_var = mat(ensemble.n_rows, ensemble.n_rows, arma::fill::eye);
        real_var *= var;
        
        mat perturb = arma::mvnrnd(vec(ensemble.n_rows,arma::fill::zeros), real_var, ensemble.n_cols);
        ensemble += perturb;
    }

    void changeNoise(mat& ensemble, const mat& sys_var){

    }
}

mat fracLorenz96model(const mat& ensemble, int idx, const mat& sys_var){
    // 从上到下时间从近到远
    int dim = config::dim;
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
    for(int i=0; i<dim; i++){
        int pre = (i + dim - 1) % dim;
        int far = (i + dim - 2) % dim;
        int next = (i + 1) % dim;

        rhs.row(i) = ensemble.row(pre) % (ensemble.row(next) - ensemble.row(far)) - ensemble.row(i) + config::F;
    }
    config::addAllTime(rhs, sys_var);

    for(int i=0; i<dim; i++){
        rhs.row(i) *= pow(config::dt, config::orders[i]);
    }

    if(window < config::window_length){
        mat ret(dim*(window+1), ensemble.n_cols, arma::fill::none);
        ret.submat(0,0,dim-1,ensemble.n_cols-1) = rhs - fracDirivative;
        ret.submat(dim,0,ret.n_rows-1,ret.n_cols-1) = ensemble;
        // if(config::sys_var_ptr)
        //     config::sys_var_ptr(ret, sys_var);
        return ret;
    }else{
        mat ret(ensemble.n_rows, ensemble.n_cols, arma::fill::none);
        ret.submat(0,0,dim-1,ensemble.n_cols-1) = rhs - fracDirivative;
        ret.submat(dim,0,ret.n_rows-1,ret.n_cols-1) = ensemble.submat(0,0,ensemble.n_rows-dim-1,ensemble.n_cols-1);
        // if(config::sys_var_ptr)
        //     config::sys_var_ptr(ret, sys_var);
        return ret;
    }
}

// we'll realise the observation operator as random matrix
// so here we omit it

mat generate_frac_lorenz96(vec v0, double F, double time_max, double dt, const mat& sys_var){
    int num_iter = time_max / dt;
    if(num_iter * dt < time_max)
        num_iter++;
    int dim = v0.n_rows;
    
    // initialize result matrix 
    mat result(num_iter+1, dim, arma::fill::none);
    for(int i=0; i<dim; i++){
        result(0, i) = v0(i);
    }

    // save config::dt, config::window_length for future recovery
    double former_dt = config::dt;
    int pre_length = config::window_length;
    config::dt = dt;
    config::window_length = INT_MAX;
    for(int i=0; i<num_iter; i++){
        v0 = fracLorenz96model(v0, i, sys_var);
        // save new simulation
        for(int j=0; j<dim; j++){
            result(i+1, j) = v0(j);
        }
    }

    // recovery config::dt
    config::dt = former_dt;
    config::window_length = pre_length;

    return result;
}

void test_lorenz96(){
    int dim = config::dim;
    vec v0 = arma::randn(dim);
    mat sys_var(dim, dim, arma::fill::zeros);
    mat sol = generate_frac_lorenz96(v0, 8, 100, 0.01, sys_var);
    sol.save("./data/lorenz96.csv", arma::raw_ascii);
}

void fracLorenz96EnKF(){
    // 参数
    int dim = config::dim;
    int ob_dim = config::ob_dim;
    double ob_var = config::ob_var;
    double sys_var = config::sys_var;
    double init_var_ = config::init_var_;
    int select_every = config::select_every;
    // ob算子
    mat H = arma::randn(ob_dim, dim);
    auto H_ob = [&H, &dim](const mat& ensemble) -> mat {
        // std::cout<<"ensemble n_rows: "<<ensemble.n_rows<<"\tn_cols: "<<ensemble.n_cols<<'\n';
        if(ensemble.n_rows == dim){
            return H * ensemble;
        }else{
            // std::cout<<"start multiplication\n";
            mat real_time = ensemble.submat(0, 0, dim-1, ensemble.n_cols-1);
            mat ret = H * real_time;
            // std::cout<<"end multiplication\n";
            return ret;
        }
    };
    // 系统误差
    auto sys_error_ptr = std::make_shared<mat>(dim, dim, arma::fill::eye);
    *sys_error_ptr *= sys_var;
    // 参考解
    vec v0 = arma::randn(dim);
    mat ref = generate_frac_lorenz96(v0, config::F, config::t_max, config::dt, *sys_error_ptr*config::real_sys_var);
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
    auto ENKFResult = StochasticENKF(ensemble_size, init_ave, init_var, ob_list, num_iter, ob_errors, ob_op, fracLorenz96model, sys_errors);
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

void fracLorenz96EnKF_version2(){
    // 参数
    int dim = config::dim;
    int ob_dim = config::ob_dim;
    double ob_var = config::ob_var;
    double sys_var = config::sys_var;
    double init_var_ = config::init_var_;
    int select_every = config::select_every;
    // ob算子
    mat H = arma::randn(ob_dim, dim);
    auto H_ob = [&H, &dim](const mat& ensemble) -> mat {
        // std::cout<<"ensemble n_rows: "<<ensemble.n_rows<<"\tn_cols: "<<ensemble.n_cols<<'\n';
        if(ensemble.n_rows == dim){
            return H * ensemble;
        }else{
            // std::cout<<"start multiplication\n";
            mat real_time = ensemble.submat(0, 0, dim-1, ensemble.n_cols-1);
            mat ret = H * real_time;
            // std::cout<<"end multiplication\n";
            return ret;
        }
    };
    // 系统误差
    auto sys_error_ptr = std::make_shared<mat>(dim, dim, arma::fill::eye);
    *sys_error_ptr *= sys_var;
    // 参考解
    vec v0 = arma::randn(dim);
    mat ref = generate_frac_lorenz96(v0, config::F, config::t_max, config::dt, *sys_error_ptr*config::real_sys_var);
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
    config::window_length = INT_MAX;
    auto ENKFResult = FStochasticENKF<40, 10>(ensemble_size, init_ave, init_var, ob_list, 
        num_iter, ob_errors, ob_op, fracLorenz96model, sys_errors, 0.1*arma::eye(dim,dim));
    std::vector<vec>& analysis_ = ENKFResult;
    // std::vector<double> skewness_ = std::get<1>(ENKFResult);
    // std::vector<double> kurtosis_ = std::get<2>(ENKFResult);
    std::cout<<"ENKF okay\n";

    // mat analysis = arma::reshape(analysis_.back(), dim, analysis_.size()).t();
    // analysis = arma::reverse(analysis);
    arma::mat analysis(analysis_.size(), analysis_[0].n_rows);
    for(int i=0; i<analysis.n_rows; i++){
        for(int j=0; j<dim; j++){
            analysis(i, j) = analysis_[i](j);
        }
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

/*
// void lorenz96particle(){
//     // 参数
//     int dim = config::dim;
//     int ob_dim = config::ob_dim;
//     double ob_var = config::ob_var;
//     double sys_var = config::sys_var;
//     double init_var_ = config::init_var_;
//     int select_every = config::select_every;

//     // 系统误差
//     auto sys_error_ptr = std::make_shared<mat>(dim, dim, arma::fill::eye);
//     *sys_error_ptr *= sys_var;

//     // 参考解
//     vec v0 = arma::randn(dim);
//     mat ref = generate_lorenz96(v0, config::F, config::t_max, config::dt, *sys_error_ptr*config::real_sys_var);
//     ref.save("./data/lorenz96.csv", arma::raw_ascii);
    
//     // ob算子
//     mat H = arma::randn(ob_dim, dim);
//     auto H_ob = [&H](const mat& ensemble){
//         return H * ensemble;
//     };
//     auto error_ptr = std::make_shared<mat>(ob_dim, ob_dim, arma::fill::eye);
//     *error_ptr *= ob_var;

//     // 辅助lambda表达式
//     auto likehood = [&H, &error_ptr](vec solid, vec ob){
//         vec ob_ = H * solid;
//         // std::cout<<"enter likehood\n";
//         vec misfit = ob_ - ob;
//         // std::cout<<"misfit:"<<misfit<<"\n";
//         mat likehood = -1./2 * misfit.t() * error_ptr->i() * misfit;
//         std::cout<<"distance:"<<likehood<<"\t";
//         std::cout<<"likehood:"<<exp(likehood(0, 0))<<"\n";
//         // std::cout<<"end likehood\n";
//         return exp(likehood(0, 0));
//     };

//     std::vector<vec> ob_list;
//     Errors ob_errors;
//     mat all_ob = H_ob(ref.t());

//     for(int i=0; i<all_ob.n_cols; i++){
//         // std::cout<<"in for\n";
//         ob_errors.add(error_ptr);
//         if(i%select_every == 0)
//             ob_list.push_back(all_ob.col(i)+mvnrnd(vec(ob_dim,arma::fill::zeros), *error_ptr));
//         else
//             ob_list.push_back(vec());
//     }

//     // 系统误差
//     Errors sys_errors;
//     for(int i=0; i<ob_list.size()+1; i++)
//         sys_errors.add(sys_error_ptr);
    
//     // 初始值
//     int size = config::ensemble_size;
//     vec init_ave(dim, arma::fill::zeros);
//     mat init_var(dim, dim, arma::fill::eye);
//     init_var *= init_var_;
//     mat ensemble = mvnrnd(init_ave, init_var, size);

//     // particle filter
//     std::cout<<"ready for particle filter\n";
//     auto particle = ParticleFilter(ensemble, lorenz96model, likehood, ob_list, sys_errors);
//     std::cout<<"particle filter ended\n";

//     // 后处理
//     mat assimilated(particle.size(), dim, arma::fill::none);

//     for(int i=0; i<particle.size(); i++){
//         vec temp(dim, arma::fill::zeros);
//         for(int j=0; j<particle[i].first.n_cols; j++){
//             temp += particle[i].first.col(j) * particle[i].second(j);
//         }
//         for(int j=0; j<dim; j++){
//             assimilated(i, j) = temp(j);
//         }
//     }
//     assimilated.save("./data/analysis.csv", arma::raw_ascii);
// }
*/

int main(int argc, char** argv){
    using namespace boost::program_options;
    using namespace config;

    int version;
    std::string problem, sys_var_type;

    options_description cmd("fractional lorenz96 EnKF");
    cmd.add_options()("problem,p", value<std::string>(&problem)->default_value("ENKF"), "type: ENKF or Particle");
    cmd.add_options()("version,e", value<int>(&version)->default_value(1), "ENKF version");
    cmd.add_options()("dim,d", value<int>(&dim)->default_value(40), "dimension of system");
    cmd.add_options()("ob_dim,o", value<int>(&ob_dim)->default_value(8), "ob dimension");
    cmd.add_options()("F,F", value<double>(&F)->default_value(8.0), "F");
    cmd.add_options()("window,w", value<int>(&window_length)->default_value(10), "window length");
    cmd.add_options()("ob_var,b", value<double>(&ob_var)->default_value(0.1), "ob_error");
    cmd.add_options()("sys_var,v", value<double>(&sys_var)->default_value(5), "system_error");
    cmd.add_options()("sys_var_type,y", value<std::string>(&sys_var_type)->default_value("real"), "system_error_type");
    cmd.add_options()("init_var,i", value<double>(&init_var_)->default_value(10), "init_error");
    cmd.add_options()("real_sys_var,r", value<double>(&real_sys_var)->default_value(1.), "real_system_error");
    cmd.add_options()("select,s", value<int>(&select_every)->default_value(10), "select every");
    cmd.add_options()("size,n", value<int>(&ensemble_size)->default_value(20), "ensemble size");
    cmd.add_options()("max_time,t", value<double>(&t_max)->default_value(20), "max time");

    variables_map map;
    store(parse_command_line(argc, argv, cmd), map);
    notify(map);

    arma::arma_rng::set_seed_random();

    if(sys_var_type == "real")
        sys_var_ptr = addRealTime;
    else if(sys_var_type == "all")
        sys_var_ptr = addAllTime;
    else if(sys_var_type == "noise")
        sys_var_ptr = changeNoise;
    else
        sys_var_ptr = nullptr;

    // lorenz63EnKF();
    if(problem == "ENKF")
        if(version == 1)
            fracLorenz96EnKF();
        else if(version == 2)
            fracLorenz96EnKF_version2();
    else
        throw std::runtime_error("not supported filter algorithm");

    std::cout<<"problem type: "<<problem<<'\n'
            <<"version: "<<version<<'\n'
            <<"window: "<<window_length<<'\n'
            <<"dim: "<<dim<<'\n'
            <<"ob_dim: "<<ob_dim<<'\n'
            <<"F: "<<F<<'\n'
            <<"init_var: "<<init_var_<<'\n'
            <<"ob_var: "<<ob_var<<'\n'
            <<"sys_var: "<<sys_var<<'\n'
            <<"sys var type: "<<sys_var_type<<'\n'
            <<"real_sys_var: "<<real_sys_var<<'\n'
            <<"select every: "<<select_every<<'\n'
            <<"ensemble size: "<<ensemble_size<<'\n'
            <<"max time: "<<t_max<<'\n'
            <<"orders: \n"<<orders<<"\n"
            <<"bino: "<<bino.submat(0,0,20-1,dim-1)<<'\n';
}