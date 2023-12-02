#include "swe.hpp"
#include "gEnKF.hpp"
#include <boost/program_options.hpp>

using namespace shiki;

void swe_test(const boost::program_options::variables_map &map)
{
    // model
    SweBHMM swe(
        map["grid_x"].as<int>(),
        map["grid_y"].as<int>(),
        map["times"].as<int>());
    swe.reference();
    std::cout << "swe.assimilate() finished" << std::endl;

    // observation setting
    LinearObserver ob(10, swe);
    int needed = ob.get_times().size();

    std::vector<arma::sp_mat> hs(needed);
    std::vector<arma::mat> noises(needed);
    RandomObserveHelper helper(swe.structure, map["ob-num"].as<int>(), map["ob-num"].as<int>(), 2);
    if (map["ob-type"].as<std::string>() == "random")
    {

        for (int i = 0; i < needed; ++i)
        {
            hs[i] = helper.generate();
            noises[i] = map["ob_var"].as<double>() * arma::eye(hs[i].n_rows, hs[i].n_rows);
        }
    }
    else
    {
        auto H = helper.generate();
        arma::mat noise = map["ob_var"].as<double>() * arma::eye(H.n_rows, H.n_rows);
        for (int i = 0; i < needed; ++i)
        {
            hs[i] = H;
            noises[i] = noise;
        }
    }
    ob.set_H_noise(hs, noises);
    ob.observe();
    std::cout << "obseravtion size:" << ob.observations.size() << std::endl;
    std::cout << "ob.observe() finished" << std::endl;

    // enkf setting
    int N = map["en-size"].as<int>();
    double init_error = map["init-error"].as<double>();

    arma::vec real0 = swe.get_state(swe.t0);
    arma::mat ensemble(real0.n_rows, N, arma::fill::none);
    for (int i = 0; i < N; ++i)
    {
        swe.init(real0, swe.structure, true, 0.3);
        ensemble.col(i) = real0;
    }
    std::cout << "start assimilation" << std::endl;

    // assimilate
    arma::vec max_error, relative_error;
    if (map["ob-type"].as<std::string>() == "random")
    {
        std::vector<shiki::GVar> gvars;
        for (int i = 0; i < 1; ++i)
        {
            gvars.push_back(GVar());
            gvars[i].k = init_error;
        }
        ggEnKF ggenkf(init_error, N, true);
        ggenkf.assimilate(swe.iter_times, swe.t0, swe.dt, ensemble, gvars, ob, swe);

        // errors
        max_error = ggenkf.max_error;
        relative_error = ggenkf.relative_error;
    }
    else
    {
        auto H = ob.linear(swe.t0, ensemble.col(0));
        arma::sp_mat init_var = init_error * arma::eye<arma::sp_mat>(real0.n_rows, real0.n_rows);
        std::vector<arma::sp_mat> vars(N, H*init_var);
        gEnKF_H genkf;
        genkf.assimilate(swe.iter_times, swe.t0, swe.dt, ensemble, vars, ob, swe);

        // errors
        max_error = genkf.max_error;
        relative_error = genkf.relative_error;
    }

    // output
    arma::mat result(relative_error.size(), 2);
    result.col(0) = max_error;
    result.col(1) = relative_error;
    result.save("./data/swe_max_rel.txt", arma::raw_ascii);
}

int main(int argc, char **argv)
{
    // use boost to parse command line
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("grid_x", po::value<int>()->default_value(150), "grid number in x direction")
        ("grid_y", po::value<int>()->default_value(150), "grid number in y direction")
        ("times", po::value<int>()->default_value(500), "iter times")
        ("ob-type", po::value<std::string>()->default_value("fixed"), "H type")
        ("ob_var", po::value<double>()->default_value(0.01), "ob var")
        ("en-size", po::value<int>()->default_value(20), "ensemble size")
        ("ob-num", po::value<int>()->default_value(11), "ob number per direction")
        ("init-error", po::value<double>()->default_value(0.03), "init error");
        
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    swe_test(vm);

    return 0;
}