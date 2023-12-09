#include "fLorenz63.hpp"
#include "3DVar.hpp"
#include "fUKF.hpp"
#include "fEnKF.hpp"
#include "fgEnKF.hpp"
#include <boost/program_options.hpp>

using namespace arma;
using namespace shiki;

void test(const boost::program_options::variables_map &map)
{
    // --------------------------Lorenz63 model------------------------------------
    arma::drowvec orders = {
        map["alpha"].as<double>(),
        map["beta"].as<double>(),
        map["gamma"].as<double>()};
    double dt = map["dt"].as<double>();
    double tmax = map["max_time"].as<double>();
    FLorenz63Model model(
        orders,
        10, 28, 8.0 / 3,
        dt, tmax,
        map["sys_var"].as<double>() * arma::eye(3, 3));
    model.set_init_state({1, 1, 1});
    model.reference();
    // save reference
    arma::mat ref(model.get_times().size(), 3);
    for (int i = 0; i < model.get_times().size(); ++i)
    {
        ref.row(i) = model.get_state(model.get_times()[i]).t();
    }
    ref.save("./data/lorenz63.csv", arma::raw_ascii);

    // -------------------------------------observation----------------------------------
    LinearObserver ob(
        map["ob_gap"].as<int>(),
        model);
    int needed = ob.get_times().size();
    std::cout << "needed:" << needed << "\n";

    // ob算子 & ob误差
    std::vector<sp_mat> hs(needed);
    std::vector<arma::mat> noises(needed);
    GeneralRandomObserveHelper helper(3, map["ob_dim"].as<int>());
    for (int i = 0; i < needed; ++i)
    {
        hs[i] = helper.generate();
        noises[i] = map["ob_var"].as<double>() * arma::eye(hs[i].n_rows, hs[i].n_rows);
    }
    ob.set_H_noise(hs, noises);
    ob.observe();

    // -------------------------------var 3d----------------------------------
    arma::vec mean = {0.9, 0.9, 0.9};
    Var3d var(true, 0.01 * arma::eye(6, 6), 2);
    var.assimilate(
        model.get_times().size(), 0, dt,
        mean, model, ob);

    // save result
    arma::mat var_result(var.res.size(), 3);
    for (int i = 0; i < var.res.size(); ++i)
    {
        var_result.row(i) = var.res[i].t();
    }
    var_result.save("./data/var_analysis.csv", arma::raw_ascii);

    // -------------------------------ukf 3d----------------------------------
    mean = {0.9, 0.9, 0.9};
    FUKF fukf(
        1, 2, 0,
        orders,
        [&](mat e)
        {
            return model.rhs(e);
        });
    arma::mat fukf_var = arma::eye(3, 3);
    fukf.assimilate(
        model.get_times().size(), 0, dt,
        mean, fukf_var, model, ob,
        0.1 * arma::eye(3, 3));
    // save result
    arma::mat fukf_result(fukf.res.size(), 3);
    for (int i = 0; i < fukf.res.size(); ++i)
    {
        fukf_result.row(i) = fukf.res[i].t();
    }
    fukf_result.save("./data/ukf_analysis.csv", arma::raw_ascii);

    // -------------------------------enkf 3d----------------------------------
    double init_var = map["init_var"].as<double>();
    int size = map["size"].as<int>();
    arma::mat ensemble = init_var * arma::randn(3, size);
    fEnKF fenkf(1);
    fenkf.assimilate(
        model.get_times().size(), 0, dt,
        ensemble, model, ob);
    // save result
    arma::mat fenkf_result(fenkf.res.size(), 3);
    for (int i = 0; i < fenkf.res.size(); ++i)
    {
        fenkf_result.row(i) = fenkf.res[i].t();
    }
    fenkf_result.save("./data/fenkf_analysis.csv", arma::raw_ascii);

    // -------------------------------fgenkf 3d----------------------------------
    fgEnKF fgenkf(3, orders);
    ensemble = 1 + 0.1 * arma::randn(3, size);
    std::vector<arma::mat> vars(size, 0.1 * arma::eye(3, 3));
    fgenkf.assimilate(
        model.get_times().size(), 0, dt, 3,
        ensemble, vars, model, ob);
    // save result
    arma::mat fgenkf_result(fgenkf.res.size(), 3);
    for (int i = 0; i < fgenkf.res.size(); ++i)
    {
        fgenkf_result.row(i) = fgenkf.res[i].t();
    }
    fgenkf_result.save("./data/fgenkf_analysis.csv", arma::raw_ascii);

}

int main(int argc, char **argv)
{
    using namespace boost::program_options;

    options_description cmd("Problem Fractional Lorenz63");
    cmd.add_options()
    ("help,h", "produce help message")
    ("alpha", value<double>()->default_value(1.1), "alpha of model")
    ("beta", value<double>()->default_value(1.1), "beta of model")
    ("gamma", value<double>()->default_value(1.1), "gamma of model")
    ("dt", value<double>()->default_value(0.005), "dt of model")
    ("max_time", value<double>()->default_value(20), "max_time of model")
    ("sys_var", value<double>()->default_value(0), "sys_var of model")
    ("ob_gap", value<int>()->default_value(10), "ob_gap of model")
    ("ob_dim", value<int>()->default_value(3), "ob_dim of model")
    ("ob_var", value<double>()->default_value(0.1), "ob_var of model")
    ("init_var", value<double>()->default_value(10), "init_var of model")
    ("size", value<int>()->default_value(20), "size of model");

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
