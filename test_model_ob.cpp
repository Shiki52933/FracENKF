#include "utility.hpp"
#include "swe.hpp"
#include "gEnKF.hpp"
using namespace shiki;
using namespace arma;
using namespace std;

int main(){
    SweBHMM swe(50, 50, 1000);
    swe.assimilate();
    cout << "swe.assimilate() finished" << endl;
    
    LinearObserver ob(10, swe);
    int needed = ob.get_times().size();
    vector<sp_mat> hs(needed, arma::sprandu(100,50*50*3, 0.1));
    vector<mat> noises(needed, 0.1*eye(100,100));
    ob.set_H_noise(hs, noises);
    ob.observe();
    cout<<"obseravtion size:" << ob.observations.size() << endl;
    cout << "ob.observe() finished" << endl;

    
    int N=10;
    ggEnKF enkf(0.1, N, false);

    vec real0 = swe.get_state(swe.t0);
    arma::mat ensemble(real0.n_rows, N, arma::fill::none);
    for(int i=0; i<N; ++i){
        swe.init(real0, swe.structure, true, 0.03);
        ensemble.col(i) = real0;
    }

    vector<shiki::GVar> gvars;
    for(int i=0; i<N; ++i){
        gvars.push_back(GVar());
        gvars[i].k = 0.03;
    }

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