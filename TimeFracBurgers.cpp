#include "StochasticENKF.hpp"
#include <stdio.h>
#include <string>

using namespace arma;

class Mesh{
    vec mMeshPoints;
    sp_mat mIntegralMat;
    sp_mat mDerivativeMat;

public:
    Mesh(vec meshPoints): mMeshPoints(meshPoints){
        int n = meshPoints.size()-2;
        mIntegralMat = sp_mat(n, n);
        mDerivativeMat = sp_mat(n,n);
        // 下面是初始化
        mIntegralMat(0,0) = (mMeshPoints[2] - mMeshPoints[0]) / 2.0;
        mIntegralMat(0,1) = (mMeshPoints[2] - mMeshPoints[1]) / 6.0;
        for(int i=1; i<n-1; i++){
            mIntegralMat(i,i-1) = (mMeshPoints[i+1] - mMeshPoints[i]) / 6.0;
            mIntegralMat(i,i) = (mMeshPoints[i+2] - mMeshPoints[i]) / 2.0;
            mIntegralMat(i,i+1) = (mMeshPoints[i+2] - mMeshPoints[i+1]) / 6.0;
        }
        mIntegralMat(n-1,n-2) = (mMeshPoints[n] - mMeshPoints[n-1]) / 6.0;
        mIntegralMat(n-1,n-1) = (mMeshPoints[n+1] - mMeshPoints[n-1]) / 2.0;
        // 下面是导数积分矩阵初始化
        mDerivativeMat(0,0) = 1.0 / (mMeshPoints[1] - mMeshPoints[0]) +
                            1.0 / (mMeshPoints[2] - mMeshPoints[1]);
        mDerivativeMat(0,1) = -1.0 / (mMeshPoints[2] - mMeshPoints[1]);
        for(int i=1; i<n-1; i++){
            mDerivativeMat(i,i-1) = -1.0/(mMeshPoints[i+1] - mMeshPoints[i]);
            mDerivativeMat(i,i) = 1.0 / (mMeshPoints[i+1] - mMeshPoints[i]) +
                                1.0/(mMeshPoints[i+2] - mMeshPoints[i+1]);
            mDerivativeMat(i,i+1) = -1.0/(mMeshPoints[i+2] - mMeshPoints[i+1]);
        }
        mDerivativeMat(n-1, n-2) = -1.0/(mMeshPoints[n] - mMeshPoints[n-1]);
        mDerivativeMat(n-1,n-1) = 1.0/(mMeshPoints[n+1] - mMeshPoints[n]) +
                                1.0 / (mMeshPoints[n] - mMeshPoints[n-1]);
    }

    const vec& getMeshPoints() {return mMeshPoints;}
    const sp_mat& getIntegralMat() {return mIntegralMat;}
    const sp_mat& getDerivativeMat() {return mDerivativeMat;}
};


class Burgers{
    double alpha;
    Mesh& mMesh;
    double mCAlphaDeltaT;
    int N;
    double v;

public:
    Burgers(double _alpha, double _v, Mesh& mesh, double deltaT)
    :alpha(_alpha), mMesh(mesh), N(mesh.getIntegralMat().n_cols), v(_v){
        mCAlphaDeltaT = pow(deltaT, -_alpha) / tgamma(2 - _alpha); 
    }

    static vec nonlinearSolver(double cAlphaDeltaT, vec f, double v, Mesh& mesh){
        const double epsilon = 1e-15;
        int count=0;

        vec b(f.size(), arma::fill::zeros);
        vec newSol(f.size(), arma::fill::zeros);
        vec oldSol(f.size(), arma::fill::ones);

        // 初始化b
        for(int j=0; j<b.size(); j++)
            for(int i= (j-1>=0?j-1:0); i<=j+1 && i<b.size(); i++)
                b[j] -= f[i]*mesh.getIntegralMat()(i,j);
        // 循环求解，直到改进很小
        while(max(abs(newSol - oldSol)) > epsilon){
            count++;
            oldSol = newSol;
            sp_mat a(f.size(), f.size());
            // 初始化a,这里j是行的下标
            for(int j=0; j<a.n_rows; j++){
                if(j-1 >= 0)
                    a(j,j-1) = cAlphaDeltaT*mesh.getIntegralMat()(j,j-1) + v*mesh.getDerivativeMat()(j,j-1) \
                                - 0.5*(oldSol[j-1]/3.0 + oldSol[j]/6.0);
                a(j,j) = cAlphaDeltaT*mesh.getIntegralMat()(j,j) + v*mesh.getDerivativeMat()(j,j) -
                            0.5*(oldSol[j-1]/6.0 - oldSol[j+1]/6.0);
                if(j+1 < a.n_cols)
                    a(j,j+1) = cAlphaDeltaT*mesh.getIntegralMat()(j,j+1) + v*mesh.getDerivativeMat()(j,j+1) -
                            0.5*(-oldSol[j]/6.0 - oldSol[j+1]/3.0);
            }
            newSol = spsolve(a, b);
        }
        // printf("计算%d次\n",count);
        // 求解完毕
        return newSol;
    }

    mat forward(mat& ensemble, int idx, mat& sysVar){
        // printf("%d\n", idx);
        int n = ensemble.n_rows / this->N;
        vec bAlpha = BAlpha(this->alpha, n+2);
        mat sols(N, ensemble.n_cols);
        
        for(int j=0; j<ensemble.n_cols; j++){
            // 约定时间近的在下面
            vec formerInfo(ensemble.submat(0,j,N-1,j));
            formerInfo *= - bAlpha[n-1];
            for(int i=1; i<n; i++){
                vec temp(ensemble.submat((n-i)*N,j,(n-i+1)*N-1,j));
                formerInfo += (bAlpha[i] - bAlpha[i-1]) * temp;
            }
            formerInfo *= this->mCAlphaDeltaT;
            vec sol = nonlinearSolver(this->mCAlphaDeltaT, formerInfo, v, mMesh);
            sols.col(j) = sol;
        }
        sols += mvnrnd(zeros(N), sysVar, ensemble.n_cols);
        return join_cols(ensemble, sols);
    }
};


vec init(const vec& x){
    return arma::sin(datum::pi * x);
}


mat burgersTest(double alpha, std::string filename){
    vec xMesh = arma::linspace(0,2,401);
    vec u0 = init(xMesh).subvec(1,399);
    mat sol(u0);
    mat sysVar(u0.size(), u0.size(), arma::fill::zeros);
    double deltaT = 0.001;
    Mesh mesh(xMesh);

    Burgers burgers(alpha, 0.01/datum::pi, mesh, deltaT);
    for(int i=0; i<(int)(1.0/deltaT); i++){
        // printf("第%d次迭代\n", i);
        sol = burgers.forward(sol,i,sysVar);
    }
    sol.reshape(xMesh.size()-2, sol.n_rows/(xMesh.size()-2));
    mat t=sol.t();
    t.save(filename, arma::raw_ascii);
    return t;
}

mat fracObOp(mat& input){
    mat obResult(4, input.n_cols);
    for(int i=0; i<4; i++)
        obResult.row(i) = input.row(input.n_rows - 80 + i*20);
    return obResult;
}

// 基本参数
double alpha = 0.8;
double v = 0.01 / arma::datum::pi;
double deltaT = 0.01;
double obVar = 0.01;
double initVar = 0.01;
int iters = 1/deltaT+1;
vec xMesh = arma::linspace(0,2,201);
Mesh mesh(xMesh);
Burgers burgers(alpha, v, mesh, deltaT);

mat wrapper(mat& ensembleAnalysis, int idx, mat& sysVar){
    //printf("第%d次\n", idx);
    return burgers.forward(ensembleAnalysis, idx, sysVar);
}

void fracBurgersENKFTest(){
    // 生成参考解
    std::string filename = std::to_string(alpha);
    filename += "_burgers.csv";
    mat sol = burgersTest(alpha, filename);
    printf("计算完参考解\n");
    
    // 填充观测值
    mat ob(4,5,arma::fill::zeros);
    for(int j=0; j<ob.n_cols; j++)
        for(int i=0; i<ob.n_rows; i++)
            ob(i,j) = sol(250*j, 80*(i+1));
    auto obErrorPtr = std::make_shared<mat>(ob.n_rows, ob.n_rows, arma::fill::eye);
    *obErrorPtr *= obVar;
    ob += arma::mvnrnd(arma::zeros(ob.n_rows), *obErrorPtr, ob.n_cols);  
    printf("ob ready\n");

    // 获得ENKF需要的参数
    // 观测列表
    std::vector<vec> obLists;
    for(int i=0; i<4; i++){
        obLists.push_back(ob.col(i));
        for(int j=0; j<24; j++)
            obLists.push_back(vec());
    }
    obLists.push_back(ob.col(4));
    printf("192 ready\n");
    // 错误矩阵
    Errors obErrors;
    for(int i=0; i<obLists.size(); i++)
        obErrors.add(obErrorPtr);
    // 观测算子
    ObserveOperator obOp = fracObOp;
    printf("199 ready\n");
    // 系统误差
    auto sysVarPtr = std::make_shared<mat>(xMesh.size()-2, xMesh.size()-2, arma::fill::eye);
    *sysVarPtr *= 0.0001;
    Errors sysErrors;
    for(int i=0; i<obLists.size(); i++)
        sysErrors.add(sysVarPtr);
    printf("206 ready\n");
    // 初始值
    vec initAve = init(xMesh.subvec(1,199));
    initAve += sqrt(initVar)*arma::randn(initAve.size());
    mat initVarMat = initVar*arma::eye(initAve.size(), initAve.size());
    // ENKF
    int ensembleSize = 20;
    printf("ready\n");

    mat forAdd(iters, initAve.size(), arma::fill::zeros);
    int numENKF=10;
    for(int i=0; i<numENKF; i++){
        printf("第%d次ENKF\n", i);
        try{
        auto result = StochasticENKF(ensembleSize, initAve, initVarMat, obLists, iters, obErrors, obOp, wrapper, sysErrors);
        auto last = result.back(); 
        mat lastMat = reshape(last, initAve.size(), last.size()/initAve.size()).t();
        forAdd += lastMat;
        }catch(std::runtime_error e){
            numENKF--;
        }
    }
    forAdd /= numENKF;
    forAdd.save("analysis"+filename, arma::raw_ascii);
}


int main(int argc, char** argv){
    fracBurgersENKFTest();
}