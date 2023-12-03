#include "EnKF.hpp"
#include "utility.hpp"
#include "gEnKF.hpp"
#include "Lorenz96.hpp"
#include <boost/program_options.hpp>

using namespace arma;
using namespace shiki;


void Lorenz96_test(const boost::program_options::variables_map &map){
    // Lorenz96 model
    Lorenz96Model model(
        map["dim"].as<int>(),
        map["F"].as<double>(),
        map["dt"].as<double>(),
        map["max_time"].as<double>(),
        map["sys_var"].as<double>() * arma::eye(map["dim"].as<int>(), map["dim"].as<int>())
    );
    model.reference();

    // observation
    LinearObserver ob(
        map["ob_gap"].as<int>(),
        model
    );
    int needed = ob.get_times().size();
    std::cout<<"needed:"<<needed<<"\n";

    // ob算子 & ob误差
    std::vector<sp_mat> hs(needed);
    std::vector<arma::mat> noises(needed);
    GeneralRandomObserveHelper helper(map["dim"].as<int>(), map["ob_dim"].as<int>());
    for(int i=0; i<needed; ++i){
        hs[i] = helper.generate();
        noises[i] = map["ob_var"].as<double>() * arma::eye(hs[i].n_rows, hs[i].n_rows);
    }
    ob.set_H_noise(hs, noises);
    ob.observe();

    // 初始值
    arma::mat ensemble;
    arma::mvnrnd(
        ensemble, 
        vec(map["dim"].as<int>()), 
        map["init_var"].as<double>() * arma::eye(map["dim"].as<int>(), map["dim"].as<int>()), 
        map["size"].as<int>());

    
    int N = map["size"].as<int>();
    // -------------------------------------enkf--------------------------------------
    EnKF enkf;
    enkf.is_linear = true;
    enkf.assimilate(
        model.get_times().size(), 0, map["dt"].as<double>(),
        ensemble, ob, model
    );
    vec re_error1(enkf.relative_error);

    enkf.is_linear = false;
    enkf.assimilate(
        model.get_times().size(), 0, map["dt"].as<double>(),
        ensemble, ob, model
    );
    vec re_error2(enkf.relative_error);

    // -----------------------------------genkf-----------------------------------------
    std::vector<arma::mat> vars;
    for(int i=0; i<N; ++i){
        vars.push_back(map["local_var"].as<double>() * arma::eye(map["dim"].as<int>(), map["dim"].as<int>()));
    }
    gEnKF genkf;
    genkf.assimilate_linear(
        model.get_times().size(), 0, map["dt"].as<double>(),
        ensemble, vars, ob, model
    );
    vec re_error3(genkf.relative_error);

    genkf.assimilate(
        model.get_times().size(), 0, map["dt"].as<double>(),
        ensemble, vars, ob, model
    );
    vec re_error4(genkf.relative_error);

    // ----------------------------------------ggenkf---------------------------------------
    std::vector<shiki::GVar> gvars;
    for(int i=0; i<N; ++i){
        gvars.push_back(GVar());
        gvars[i].k = 10;
    }

    ggEnKF ggenkf(0.1, N, true);
    // 注意ggenkf中用clean_up()清空了relative_error
    ggenkf.assimilate(model.get_times().size(), 0, map["dt"].as<double>(), ensemble, gvars, ob, model);
    vec re_error5(ggenkf.relative_error);

    // output
    arma::mat result(re_error1.size(), 5);
    result.col(0) = re_error1;
    result.col(1) = re_error2;
    result.col(2) = re_error3;
    result.col(3) = re_error4;
    result.col(4) = re_error5;
    result.save("./data/lorenz96.csv", arma::raw_ascii);
}


int main(int argc, char** argv){
    using namespace boost::program_options;

    options_description cmd("Problem Lorenz96");
    cmd.add_options()("problem,p", value<std::string>()->default_value("ENKF"));
    cmd.add_options()("dim,d", value<int>()->default_value(40), "dimension of system");
    cmd.add_options()("ob_dim,o", value<int>()->default_value(40), "ob dimension");
    cmd.add_options()("F,F", value<double>()->default_value(8.0), "F");
    cmd.add_options()("dt", value<double>()->default_value(0.005), "delta t");
    cmd.add_options()("ob_var,b", value<double>()->default_value(0.1), "ob_error");
    cmd.add_options()("sys_var,v", value<double>()->default_value(1), "system_error");
    cmd.add_options()("init_var,i", value<double>()->default_value(10), "init_error");
    cmd.add_options()("local_var,l", value<double>()->default_value(0.1), "local_var for gENKF");
    cmd.add_options()("ob_gap", value<int>()->default_value(10), "ob every");
    cmd.add_options()("size,n", value<int>()->default_value(20), "ensemble size");
    cmd.add_options()("max_time,t", value<double>()->default_value(20), "max time");
    cmd.add_options()("seed", value<int>()->default_value(6666), "random seed");

    variables_map map;
    store(parse_command_line(argc, argv, cmd), map);
    notify(map);

    arma::arma_rng::set_seed_random();

    Lorenz96_test(map);
    
}