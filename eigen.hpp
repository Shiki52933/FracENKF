#pragma once
#include "utility.hpp"

namespace shiki{

arma::vec dot_vec(double k, const std::vector<arma::mat> &A, const std::vector<arma::mat> &B,const arma::vec &v) {
    arma::vec result = k * v;
    for (int i = 0; i < A.size(); i++) {
        result += A[i] * (B[i].t() * v);
    }
    return result;
}

auto biggest_eigen_pow(double k, const std::vector<arma::mat>& A, const std::vector<arma::mat>& B) {
    arma::vec eigen_vector = arma::randu<arma::vec>(A[0].n_rows);
    eigen_vector = arma::normalise(eigen_vector);

    double eigen_value = 0;
    double error = 1;
    int i=0;
    int iterations = 1000;
    double tolerance = 1e-8;

    // optimize the iterations
    arma::vec next = dot_vec(k, A, B, eigen_vector);
    
    while(i++ < iterations and error > tolerance){
        // std::cout<<"iteration: " << i << " eigen value: "<< eigen_value << " error: " << error << std::endl;
        eigen_vector = next;
        eigen_vector = arma::normalise(eigen_vector);
        next = dot_vec(k, A, B, eigen_vector);
        eigen_value = arma::dot(eigen_vector, next);
        error = arma::norm(next - eigen_value * eigen_vector);
    }

    return std::make_tuple(i, eigen_value, eigen_vector);
}


auto eigen_pow(double k, std::vector<arma::mat>& A, std::vector<arma::mat>& B, int N=-1){
    auto num = A[0].n_rows;
    auto former_len = A.size();
    if(N == -1){
        N = num;
    }

    if(num < 1000){
        arma::mat total = k * arma::eye(num, num);
        for(int i=0; i<former_len; ++i){
            total += A[i] * B[i].t();
        }
        // use armadillo to compute eigen values and vectors
        arma::vec eigval;
        arma::mat eigvec;
        arma::eig_sym(eigval, eigvec, total);
        // sort the eigen values and vectors
        arma::uvec sorted_index = arma::sort_index(eigval, "descend");
        eigval = eigval(sorted_index);
        eigvec = eigvec.cols(sorted_index);
        eigval = eigval.subvec(0, N-1);
        eigvec = eigvec.cols(0, N-1);
        return std::make_tuple(eigval, eigvec);
    }

    // use power method to compute eigen values and vectors
    arma::vec eigen_values(N);
    arma::mat eigen_vectors(num, N);

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
inline auto decompose_to_Q(arma::mat &Q, const arma::mat &A){
    std::vector<arma::vec> R(A.n_cols);
    for(int i=0; i<A.n_cols; ++i){
        arma::vec v = A.col(i);
        std::vector<double> coef;
        for(int j=0; j<Q.n_cols and arma::norm(v) > 1e-8; ++j){
            double c = arma::dot(v, Q.col(j));
            coef.push_back(c);
            v -= c * Q.col(j);
        }
        double res_norm = arma::norm(v);
        if(res_norm > 1e-8){
            // add new column to Q
            Q = arma::join_rows(Q, v/res_norm);
            coef.push_back(res_norm);
        }
        R[i] = arma::vec(coef);
    }
    // resize R and concatenate
    assert (R.size() == A.n_cols);
    for(arma::vec &v: R){
        v.resize(Q.n_cols);
    }
    arma::mat result(Q.n_cols, A.n_cols, arma::fill::none);
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
auto eigen_pow_opt(double k, const std::vector<arma::mat>& A, const std::vector<arma::mat>& B, int N=-1){
    auto num = A[0].n_rows;
    auto former_len = A.size();
    if(N == -1){
        N = num;
    }

    assert (A.size() > 0);

    // decompose 
    arma::mat q,r;
    arma::qr_econ(q, r, A[0]);
    arma::mat r_1 = decompose_to_Q(q, B[0]);
    r = r * r_1.t();

    for(int i=1; i<A.size(); ++i){
        arma::mat r_a = decompose_to_Q(q, A[i]);
        arma::mat r_b = decompose_to_Q(q, B[i]);
        arma::mat r_i = r_a * r_b.t() ;
        // std::cout<<r.n_rows << ' '<< r.n_cols<<std::endl;
        // std::cout<<r_i.n_rows << ' '<< r_i.n_cols<<std::endl;
        r_i.submat(0, 0, r.n_rows-1, r.n_cols-1) += r;
        r = r_i;
    }

    auto size1 = r.n_rows;
    auto size2 = r.n_cols;
    r.resize(q.n_cols, q.n_cols);
    // std::cout<<"size of r: "<<r.n_rows<<std::endl;
    // check extra data is 0 and r is symmetric
    // for(int i=size1; i<r.n_rows; ++i){
    //     for(int j=size2; j<r.n_cols; ++j){
    //         assert(r(i, j) == 0);
    //     }
    // }
    assert (r.is_symmetric(1e-4));
    
    // decompose r to Q diag Q^T
    arma::vec eigen_values;
    arma::mat eigen_vectors;
    arma::eig_sym(eigen_values, eigen_vectors, r);
    // sort the eigen values and vectors
    arma::uvec sorted_index = arma::sort_index(eigen_values, "descend");
    eigen_values = eigen_values(sorted_index);
    eigen_vectors = eigen_vectors.cols(sorted_index);

    q = q * eigen_vectors;

    N = std::min(N, (int)eigen_values.n_elem);
    eigen_values = eigen_values.subvec(0, N-1) + k;
    eigen_vectors = q.cols(0, N-1);

    return std::make_tuple(eigen_values, eigen_vectors);
}


}