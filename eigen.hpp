#include "StochasticENKF.hpp"

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
        std::cout<<"iteration: " << i << " eigen value: "<< eigen_value << " error: " << error << std::endl;
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
}