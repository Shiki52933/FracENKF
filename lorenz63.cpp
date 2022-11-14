#include "StochasticENKF.hpp"
#include <armadillo>


mat generate_lorenz63(double sigma, double rho, double beta, double dt, double t_max, vec x0){
    int num_iter = t_max / dt;
    if( dt * num_iter < t_max )
        num_iter++;

    mat sol(3, num_iter+1);

    sol.col(0) = x0;
    for(int i=0; i<num_iter; i++){
        double x_old = sol.col(i)[0];
        double y_old = sol.col(i)[1];
        double z_old = sol.col(i)[2];
        double dt_ = (i+1)*dt <= t_max ? dt : t_max - (i+1)*dt;

        double x_new = sigma * (y_old - x_old) * dt_ + x_old;
        double y_new = ( x_old * (rho - z_old) - y_old ) * dt_ + y_old;
        double z_new = ( x_old * y_old - beta * z_old ) * dt_ + z_old;

        sol.col(i+1)[0] = x_new;
        sol.col(i+1)[1] = y_new;
        sol.col(i+1)[2] = z_new;
    } 
    sol = sol.t();
    sol.save("lorenz63.csv", arma::raw_ascii);
    return sol;
}

mat H_ob(mat& ensemble){
    vec a{0, 1, 1};
    vec b{1, 1, 0};
    mat H(2, 3);
    H.row(0) = a;
    H.row(1) = b;
    return H * ensemble;
}

mat model(mat& ensemble, int idx, mat& sys_var){
    double sigma = 10, rho = 28, beta = 8.0/3;
    double dt = 0.001;
    mat sol = ensemble;
    for(int i=0; i<ensemble.n_cols; i++){
        double x_old = sol.col(i)[0];
        double y_old = sol.col(i)[1];
        double z_old = sol.col(i)[2];
        double dt_ = dt;

        double x_new = sigma * (y_old - x_old) * dt_ + x_old;
        double y_new = ( x_old * (rho - z_old) - y_old ) * dt_ + y_old;
        double z_new = ( x_old * y_old - beta * z_old ) * dt_ + z_old;

        sol.col(i)[0] = x_new;
        sol.col(i)[1] = y_new;
        sol.col(i)[2] = z_new;
    }
    return sol;
}

void lorenz63EnKF(){

}

int main(int argc, char** argv){
    generate_lorenz63(10., 28., 8.0/3, 0.001, 200, vec{1,1,1});
}