#pragma once
#include <armadillo>

namespace shiki{

arma::vec Krylov(
    const arma::sp_mat& A, 
    const arma::vec& b, 
    double tol=1e-4
    ){
    int n = A.n_rows;
    int m=2;
    arma::vec current = b;
    arma::mat approx = arma::mat(A.n_rows, m, arma::fill::none);
    for(int j=0; j<m; ++j){
        approx.col(j) = current;
        current = A * current;
    }

    arma::vec beta = arma::solve(approx.submat(0,1,n-1,m-1), b);
    arma::vec x(n, arma::fill::zeros);
    for(int j=0; j<m-1; ++j){
        x += beta(j) * approx.col(j);
    }
    double error = arma::norm(b - A * x);

    while(error > tol){
        int next_m = 10 * (m-1) + 1;
        approx.resize(n, next_m);
        for(int j=m; j<next_m; ++j){
            approx.col(j) = current;
            current = A * current;
        }
        m = next_m;
        beta = arma::solve(approx.submat(0,1,n-1,m-1), b);
        x = arma::vec(n, arma::fill::zeros);
        for(int j=0; j<m-1; ++j){
            x += beta(j) * approx.col(j);
        }
        error = arma::norm(b - A * x);
    } 
    
    // std::cout<<"Krylov method converged in "<<m<<" iterations."<<std::endl;
    return x;
}

}