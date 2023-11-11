#pragma once
#include "StochasticENKF.hpp"
#include <assert.h>

namespace shiki{
    using std::vector, std::make_tuple;

vec dot_vec(double k, const vector<mat> &A, const vector<mat> &B,const vec &v) {
    vec result = k * v;
    for (int i = 0; i < A.size(); i++) {
        result += A[i] * (B[i].t() * v);
    }
    return result;
}

auto biggest_eigen_pow(double k, const vector<mat>& A, const vector<mat>& B) {
    vec eigen_vector = arma::randu<vec>(A[0].n_rows);
    eigen_vector = normalise(eigen_vector);

    double eigen_value = 0;
    double error = 1;
    int i=0;
    int iterations = 1000;
    double tolerance = 1e-8;

    // optimize the iterations
    vec next = dot_vec(k, A, B, eigen_vector);
    
    while(i++ < iterations and error > tolerance){
        // std::cout<<"iteration: " << i << " eigen value: "<< eigen_value << " error: " << error << std::endl;
        eigen_vector = next;
        eigen_vector = normalise(eigen_vector);
        next = dot_vec(k, A, B, eigen_vector);
        eigen_value = dot(eigen_vector, next);
        error = arma::norm(next - eigen_value * eigen_vector);
    }

    return std::make_tuple(i, eigen_value, eigen_vector);
}


auto eigen_pow(double k, vector<mat>& A, vector<mat>& B, int N=-1){
    auto num = A[0].n_rows;
    auto former_len = A.size();
    if(N == -1){
        N = num;
    }

    if(num < 1000){
        mat total = k * arma::eye(num, num);
        for(int i=0; i<former_len; ++i){
            total += A[i] * B[i].t();
        }
        // use armadillo to compute eigen values and vectors
        vec eigval;
        mat eigvec;
        eig_sym(eigval, eigvec, total);
        // sort the eigen values and vectors
        arma::uvec sorted_index = sort_index(eigval, "descend");
        eigval = eigval(sorted_index);
        eigvec = eigvec.cols(sorted_index);
        eigval = eigval.subvec(0, N-1);
        eigvec = eigvec.cols(0, N-1);
        return std::make_tuple(eigval, eigvec);
    }

    // use power method to compute eigen values and vectors
    vec eigen_values(N);
    mat eigen_vectors(num, N);

    for(int i=0; i<N; ++i){
        auto [iter, eigen_value, eigen_vector] = biggest_eigen_pow(k, A, B);
        eigen_values(i) = eigen_value;
        eigen_vectors.col(i) = eigen_vector;
        // remove the biggest eigen value and vector
        A.push_back(-eigen_value * eigen_vector);
        B.push_back(eigen_vector);
    }

    // remove appended matrices
    A.erase(A.begin() + former_len, A.end());
    B.erase(B.begin() + former_len, B.end());

    return std::make_tuple(eigen_values, eigen_vectors);
}


/// @brief decompose A = (Q, Q_1)R, where A, Q are given and (Q, Q_1) is orthogonal
/// @param Q 
/// @param A 
/// @return R. but Q is modified
inline auto decompose_to_Q(mat &Q, const mat &A){
    vector<vec> R(A.n_cols);
    for(int i=0; i<A.n_cols; ++i){
        vec v = A.col(i);
        vector<double> coef;
        for(int j=0; j<Q.n_cols and norm(v) > 1e-8; ++j){
            double c = dot(v, Q.col(j));
            coef.push_back(c);
            v -= c * Q.col(j);
        }
        double res_norm = norm(v);
        if(res_norm > 1e-8){
            // add new column to Q
            Q = join_rows(Q, v/res_norm);
            coef.push_back(res_norm);
        }
        R[i] = vec(coef);
    }
    // resize R and concatenate
    assert (R.size() == A.n_cols);
    for(vec &v: R){
        v.resize(Q.n_cols);
    }
    mat result(Q.n_cols, A.n_cols, arma::fill::none);
    for(int i=0; i<A.n_cols; ++i){
        result.col(i) = R[i];
    }

    return result;
} 


/// @brief compute the eigen values and vectors of the matrix A = k * I + sum_{i=1}^m A_i * B_i^T
/// @brief Assumption: A_i is of size L*S, B_i is of size L*S, where L >> S
/// @brief Assumption: A - kI is sysmetric
/// @brief principle: write A_i = QR_i, B_i = QS_i, where Q is orthogonal and independant of k
/// @brief then A = kI + sum_{i=1}^m QR_i S_i^T Q^T = kI + Q sum_{i=1}^m R_i S_i^T Q^T = kI + Q R Q^T
/// @brief then we write R = Q_1 C Q_1^T, where Q_1 is orthogonal and C is diagonal
/// @brief if we let Q = QQ_1, then A = kI + Q C Q^T = kQQ^T + Q C Q^T = Q (kI + C) Q^T,
/// @brief if values in C is ordered in descending order, then the first N eigen values and vectors are clear.
/// @brief we hope this algorithm is faster than the naive one
/// @param k > 0 the coefficient of the identity matrix
/// @param A a vector of matrices
/// @param B a vector of matrices
/// @param N the number of eigen values and vectors to be computed
/// @return a tuple of eigen values and vectors
auto eigen_pow_opt(double k, const vector<mat>& A, const vector<mat>& B, int N=-1){
    auto num = A[0].n_rows;
    auto former_len = A.size();
    if(N == -1){
        N = num;
    }

    assert (A.size() > 0);

    // decompose 
    mat q,r;
    arma::qr_econ(q, r, A[0]);
    mat r_1 = decompose_to_Q(q, B[0]);
    r = r * r_1.t();

    for(int i=1; i<A.size(); ++i){
        mat r_a = decompose_to_Q(q, A[i]);
        mat r_b = decompose_to_Q(q, B[i]);
        mat r_i = r_a * r_b.t() ;
        // std::cout<<r.n_rows << ' '<< r.n_cols<<std::endl;
        // std::cout<<r_i.n_rows << ' '<< r_i.n_cols<<std::endl;
        r_i.submat(0, 0, r.n_rows-1, r.n_cols-1) += r;
        r = r_i;
    }

    auto size1 = r.n_rows;
    auto size2 = r.n_cols;
    r.resize(q.n_cols, q.n_cols);
    std::cout<<"size of r: "<<r.n_rows<<std::endl;
    // check extra data is 0 and r is symmetric
    // for(int i=size1; i<r.n_rows; ++i){
    //     for(int j=size2; j<r.n_cols; ++j){
    //         assert(r(i, j) == 0);
    //     }
    // }
    assert (r.is_symmetric(1e-4));
    
    // decompose r to Q diag Q^T
    vec eigen_values;
    mat eigen_vectors;
    eig_sym(eigen_values, eigen_vectors, r);
    // sort the eigen values and vectors
    arma::uvec sorted_index = sort_index(eigen_values, "descend");
    eigen_values = eigen_values(sorted_index);
    eigen_vectors = eigen_vectors.cols(sorted_index);

    q = q * eigen_vectors;

    N = std::min(N, (int)eigen_values.n_elem);
    eigen_values = eigen_values.subvec(0, N-1) + k;
    eigen_vectors = q.cols(0, N-1);

    return std::make_tuple(eigen_values, eigen_vectors);
}


}