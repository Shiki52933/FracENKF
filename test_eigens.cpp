// test eigen values and vectors
// check if power and armadillo give the same results
//
// Path: test/test_eigens.cpp
// Compare this snippet from eigen.cpp:
//
#include "eigen.hpp"
using namespace std;
using namespace arma;
using namespace shiki;

int main(){
    auto sizes = {2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,};
    for(int size: sizes){
        for(int i=0; i<5; ++i){
            // generate random matrices
            vector<mat> A;
            vector<mat> B;
            for(int i=0; i<10; ++i){
                mat t = arma::randu<mat>(size, 20);
                A.push_back(t);
                B.push_back(t);
            }
            // compute eigen values and vectors
            auto time0 = std::chrono::high_resolution_clock::now();
            auto [eigval, eigvec] = eigen_pow(1, A, B,20);
            auto time1 = std::chrono::high_resolution_clock::now();
            auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>( time1 - time0 ).count();

            // // check if power and armadillo give the same results
            // mat total = arma::eye(size, size);
            // for(int i=0; i<A.size(); ++i){
            //     total += A[i] * B[i].t();
            //     // cout<<A[i]<<endl;
            //     // cout<<B[i]<<endl;
            // }
            // // cout<<total<<endl;
            // vec eigval2;
            // mat eigvec2;

            // auto time2 = std::chrono::high_resolution_clock::now();
            // eig_sym(eigval2, eigvec2, total);
            // auto time3 = std::chrono::high_resolution_clock::now();
            // auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>( time3 - time2 ).count();
            // // sort the eigen values and vectors
            // arma::uvec sorted_index = sort_index(eigval2, "descend");
            // eigval2 = eigval2(sorted_index);
            // eigvec2 = eigvec2.cols(sorted_index);
            // eigval2 = eigval2.subvec(0, 19);

            // test eigen_pow_opt
            auto time4 = std::chrono::high_resolution_clock::now();
            auto [eigval3, eigvec3] = eigen_pow_opt(1, A, B,20);
            auto time5 = std::chrono::high_resolution_clock::now();
            auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>( time5 - time4 ).count();


            if(arma::max(arma::abs(eigval - eigval3)) > 1){
                std::cout << "eigen values are not the same" << std::endl;
                // print the eigen values one by one
                for(int i=0; i<eigval.size(); ++i){
                    std::cout << eigval(i) << " " << " " << eigval3(i) << std::endl;
                }
                // return 1;
            }else{
                std::cout << "size : " << size << " test: " << i << " eigen values are the same" << std::endl;
                // std::cout << "eigen values time: " << duration0 << " us" << std::endl;
                // std::cout << "armadillo time: " << duration1 << " us" << std::endl;
                // std::cout << "ratio: " << (double)duration0 / duration1 << std::endl;
                // std::cout << "eigen_pow_opt time: " << duration2 << " us" << std::endl;
            }
                std::cout << "ratio of opt: " << (double)duration2 / duration0 << std::endl;
            
        }
    }
}
