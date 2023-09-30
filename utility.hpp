#pragma once
#include <armadillo>
#include "StochasticENKF.hpp"
#include "Krylov.hpp"

namespace shiki{
    

class GVar{
public:
    double k;
    std::vector<mat> A, B;
};

class BigHiddenMarkovModel{
    public:
    virtual void assimilate() = 0;
    virtual vec get_state(double) = 0;
    virtual std::vector<double> get_times() = 0;
    virtual mat model(double t, double dt, const mat & e) = 0;
    virtual sp_mat noise(double) = 0;
};

class Observer{
    public:
    virtual bool is_observable(double) = 0;
    virtual void observe() = 0;
    virtual vec get_observation(double) = 0;
    virtual vec observe(double, vec) = 0;
    virtual mat observe(double, mat) = 0;
    virtual sp_mat linear(double, vec) = 0;
    virtual mat noise(double) = 0;
};

    using BHMM = BigHiddenMarkovModel;

}