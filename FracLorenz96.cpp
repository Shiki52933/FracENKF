#include "fLorenz96.hpp"
#include "3DVar.hpp"
#include "fUKF.hpp"
#include "fEnKF.hpp"
#include "fgEnKF.hpp"
#include <boost/program_options.hpp>

using namespace arma;
using namespace shiki;

void test(const boost::program_options::variables_map &map)
{
    // --------------------------Lorenz96 model------------------------------------
    int dim = map["dim"].as<int>();
    double F = map["F"].as<double>();
    double dt = map["dt"].as<double>();
    double tmax = map["max_time"].as<double>();
    FLornez96Model model(
        dim, F, dt, tmax,
        map["sys_var"].as<double>() * arma::eye(dim, dim));
    model.reference();
    // save reference
    arma::mat ref(model.get_times().size(), dim);
    for (int i = 0; i < model.get_times().size(); ++i)
    {
        ref.row(i) = model.get_state(model.get_times()[i]).t();
    }
    ref.save("./data/lorenz96.csv", arma::raw_ascii);

    // -------------------------------------observation----------------------------------
    LinearObserver ob(
        map["ob_gap"].as<int>(),
        model);
    int needed = ob.get_times().size();
    std::cout << "needed:" << needed << "\n";

    // ob算子 & ob误差
    std::vector<sp_mat> hs(needed);
    std::vector<arma::mat> noises(needed);
    GeneralRandomObserveHelper helper(dim, map["ob_dim"].as<int>());
    for (int i = 0; i < needed; ++i)
    {
        hs[i] = helper.generate();
        noises[i] = map["ob_var"].as<double>() * arma::eye(hs[i].n_rows, hs[i].n_rows);
    }
    ob.set_H_noise(hs, noises);
    ob.observe();

    // -------------------------------var 3d----------------------------------
    arma::vec mean = arma::randn(dim);
    Var3d var(true, 0.01 * arma::eye(2*dim, 2*dim), 2);
    std::cout<<"---------------------------var 3d----------------------------"<<std::endl;
    var.assimilate(
        model.get_times().size(), 0, dt,
        mean, model, ob);

    // save result
    arma::mat var_result(var.res.size(), dim);
    for (int i = 0; i < var.res.size(); ++i)
    {
        var_result.row(i) = var.res[i].t();
    }
    var_result.save("./data/var_analysis.csv", arma::raw_ascii);

    // -------------------------------ukf 3d----------------------------------
    // FUKF fukf(
    //     1, 2, 0,
    //     model.orders,
    //     [&](mat e){ return model.rhs(e);});
    // arma::mat fukf_var = arma::eye(dim, dim);
    // std::cout<<"---------------------------ukf----------------------------"<<std::endl;
    // fukf.assimilate(
    //     model.get_times().size(), 0, dt,
    //     mean, fukf_var, model, ob,
    //     0.1 * arma::eye(dim, dim));
    // // save result
    // arma::mat fukf_result(fukf.res.size(), dim);
    // for (int i = 0; i < fukf.res.size(); ++i)
    // {
    //     fukf_result.row(i) = fukf.res[i].t();
    // }
    // fukf_result.save("./data/ukf_analysis.csv", arma::raw_ascii);

    // -------------------------------enkf 3d----------------------------------
    double init_var = map["init_var"].as<double>();
    int size = map["size"].as<int>();
    arma::mat ensemble = init_var * arma::randn(dim, size);
    fEnKF fenkf(5);
    std::cout<<"---------------------------enkf----------------------------"<<std::endl;
    fenkf.assimilate(
        model.get_times().size(), 0, dt,
        ensemble, model, ob);
    // save result
    arma::mat fenkf_result(fenkf.res.size(), dim);
    for (int i = 0; i < fenkf.res.size(); ++i)
    {
        fenkf_result.row(i) = fenkf.res[i].t();
    }
    fenkf_result.save("./data/fenkf_analysis.csv", arma::raw_ascii);

    // ------------------------------------fgenkf----------------------------------
    fgEnKF fgenkf(3, model.orders);
    ensemble.each_col() = model.get_state(0);
    std::vector<arma::mat> vars(size, 0.1 * arma::eye(dim, dim));
    std::cout<<"---------------------------fgenkf----------------------------"<<std::endl;
    fgenkf.assimilate(
        model.get_times().size(), 0, dt, dim,
        ensemble, vars, model, ob);
    // save result
    arma::mat fgenkf_result(fgenkf.res.size(), dim);
    for (int i = 0; i < fgenkf.res.size(); ++i)
    {
        fgenkf_result.row(i) = fgenkf.res[i].t();
    }
    fgenkf_result.save("./data/fgenkf_analysis.csv", arma::raw_ascii); 
}

int main(int argc, char **argv)
{
    using namespace boost::program_options;

    options_description cmd("Problem Fractional Lorenz96");
    cmd.add_options()
        ("help,h", "produce help message")
        ("dim,d", value<int>()->default_value(40), "dimension of the model")
        ("F", value<double>()->default_value(8), "F of the model")
        ("dt", value<double>()->default_value(0.01), "dt of the model")
        ("max_time", value<double>()->default_value(20), "max time of the model")
        ("sys_var", value<double>()->default_value(1), "variance of the model")
        ("ob_gap", value<int>()->default_value(10), "gap of the observation")
        ("ob_dim", value<int>()->default_value(20), "dimension of the observation")
        ("ob_var", value<double>()->default_value(0.1), "variance of the observation")
        ("init_var", value<double>()->default_value(10), "variance of the initial ensemble")
        ("size", value<int>()->default_value(20), "size of the initial ensemble")
        ;
    variables_map map;
    store(parse_command_line(argc, argv, cmd), map);
    notify(map);

    if (map.count("help"))
    {
        std::cout << cmd << "\n";
        return 0;
    }

    test(map);

    return 0;
}
