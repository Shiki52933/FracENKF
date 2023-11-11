#include "utility.hpp"
#include "swe.hpp"
#include "gEnKF.hpp"
using namespace shiki;
using namespace arma;
using namespace std;

int main(){
    // reference model
    SweBHMM swe(100, 100, 500);
    swe.reference();
    cout << "swe.assimilate() finished" << endl;
    
    // observation setting
    LinearObserver ob(10, swe);
    int needed = ob.get_times().size();

    vector<sp_mat> hs(needed);
    RandomObserveHelper helper(swe.structure, 10, 10, 2);
    for(int i=0; i<needed; ++i){
        hs[i] = helper.generate();
    }

    vector<mat> noises(needed);
    for(int i=0; i<needed; ++i){
        noises[i] = 0.01 * eye(hs[i].n_rows, hs[i].n_rows);
    }
    ob.set_H_noise(hs, noises);
    ob.observe();
    cout<<"obseravtion size:" << ob.observations.size() << endl;
    cout << "ob.observe() finished" << endl;

    // enkf setting
    int N=20;
    ggEnKF enkf(0.1, N, true);

    vec real0 = swe.get_state(swe.t0);
    arma::mat ensemble(real0.n_rows, N, arma::fill::none);
    for(int i=0; i<N; ++i){
        swe.init(real0, swe.structure, false, 0.03);
        ensemble.col(i) = real0;
    }

    vector<shiki::GVar> gvars;
    for(int i=0; i<1; ++i){
        gvars.push_back(GVar());
        gvars[i].k = 0.03;
    }

    std::cout<<"start assimilation" << std::endl;

    enkf.assimilate(swe.iter_times, swe.t0, swe.dt, ensemble, gvars, ob, swe);

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