#include "utility.hpp"
#include "Lorenz96.hpp"
#include "swe.hpp"
#include "gEnKF.hpp"
using namespace shiki;
using namespace arma;
using namespace std;

int main(){
    // reference model
    int dim=40, obs_num = 20;
    double t_max = 20, dt = 0.005;
    Lorenz96Model l96(dim, 8, dt, t_max, 0. * eye(40, 40));
    l96.assimilate();
    cout << "l96.assimilate() finished" << endl;
    
    // observation setting
    LinearObserver ob(10, l96);
    int needed = ob.get_times().size();

    vector<sp_mat> hs(needed);
    GeneralRandomObserveHelper helper(dim, obs_num);
    for(int i=0; i<needed; ++i){
        hs[i] = helper.generate();
    }

    vector<mat> noises(needed);
    for(int i=0; i<needed; ++i){
        noises[i] = 0.1 * eye(hs[i].n_rows, hs[i].n_rows);
    }
    ob.set_H_noise(hs, noises);
    ob.observe();
    cout<<"obseravtion size:" << ob.observations.size() << endl;
    cout << "ob.observe() finished" << endl;

    // enkf setting
    int N=20;
    ggEnKF enkf(0.1, N, true);

    arma::mat ensemble = mvnrnd(vec(dim), 10*eye(dim, dim), N);

    vector<shiki::GVar> gvars;
    for(int i=0; i<N; ++i){
        gvars.push_back(GVar());
        gvars[i].k = 10;
    }

    enkf.assimilate(l96.get_times().size(), 0, dt, ensemble, gvars, ob, l96);

    // print the errors saved in enkf
    cout<< "errors: " << endl;
    for(auto error: enkf.errors){
        cout << error << endl;
    }

    cout<< "ratios: " << endl;
    for(auto ratio: enkf.eigen_ratios){
        cout << ratio << endl;
    }

}