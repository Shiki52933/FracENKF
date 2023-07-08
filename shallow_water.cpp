#include "StochasticENKF.hpp"
#include "gEnKF.hpp"
#include <limits>
#include <cmath>
#include <assert.h>

using namespace shiki;

double quadratic_approximation(
    double t1, double f1,
    double t2, double f2,
    double t3, double f3
    ){
    // return minmum of quadratic approximation
    arma::mat A = {{t1*t1, t1, 1},
                   {t2*t2, t2, 1},
                   {t3*t3, t3, 1}};
    arma::vec b = {f1, f2, f3};
    arma::vec x = arma::solve(A, b);

    if(x[0] > 0){
        return -x[1]/(2*x[0]);
    }else{
        return std::numeric_limits<double>::infinity();
    }
}

const double g = 2;

enum class BoundaryCondition{
    Periodic,
    Reflective,
    Open
};


class Structure2d{
    // 我们的坐标轴原点在左上角，i是行，从上往下，j是列，从左往右
    // 所以遍历时是从左往右，从上往下
public:
    double m_left, m_right;
    double m_low, m_high; 
    int m_grid_x, m_grid_y, m_unknowns;
    BoundaryCondition m_bc;
    double m_dx, m_dy;

public:
    Structure2d(
        double left, 
        double right, 
        int grid_x, 
        double low, 
        double high, 
        int grid_y, 
        int unknowns,
        BoundaryCondition bc=BoundaryCondition::Open
    )
    // init class members
    : m_left(left), m_right(right), m_low(low), m_high(high), m_grid_x(grid_x), m_grid_y(grid_y), m_unknowns(unknowns), m_bc(bc)
    {
        m_dx = (m_right - m_left) / (m_grid_x - 1);
        m_dy = (m_high - m_low) / (m_grid_y - 1);

        int correction = 0;
        if(m_bc == BoundaryCondition::Periodic){
            correction = -1;
        }
        m_grid_x += correction;
        m_grid_y += correction;
    }

    void allocate_fields(arma::vec &fields){
        fields.set_size(m_grid_x*m_grid_y*m_unknowns);
    }
    
    // overload () operator to access fields
    // i: x index
    // j: y index
    // k: field index
    // x axis is flipped
    inline double& operator()(arma::vec &fields, int i, int j, int k){
        return fields(k*m_grid_y*m_grid_x + j*m_grid_y + i);
    }

    // overload () operator to set jacobian
    inline void operator()(
        arma::sp_mat& jacobian,
        int fi, int fj, int fk,
        int yi, int yj, int yk,
        double value
        ){
        jacobian(fk*m_grid_y*m_grid_x + fj*m_grid_y + fi, yk*m_grid_y*m_grid_x + yj*m_grid_y + yi) = value;
        }
    
    inline auto operator()(
        arma::sp_mat& jacobian,
        int fi, int fj, int fk,
        int yi, int yj, int yk
        ){
        return jacobian(fk*m_grid_y*m_grid_x + fj*m_grid_y + fi, yk*m_grid_y*m_grid_x + yj*m_grid_y + yi);
        }
};


arma::vec form_Ifunction(
    double t, 
    arma::vec& Y,
    arma::vec& Ydot,
    Structure2d& s
){
    double dx = s.m_dx;
    double dy = s.m_dy;

    using std::pow;
    arma::vec i_left(Y.n_rows, arma::fill::none);

    for(int j=0; j<s.m_grid_x; j++){
        for(int i=0; i<s.m_grid_y; i++){
            if(i == 0 or i == s.m_grid_y-1 or j == 0 or j == s.m_grid_x-1){
                s(i_left,i,j,0) = s(Y,i,j,0);
                s(i_left,i,j,1) = s(Y,i,j,1);
                s(i_left,i,j,2) = s(Y,i,j,2)-1;
                continue;
            }

            s(i_left,i,j,0) = s(Y,i,j,0)*s(Ydot,i,j,2) + s(Y,i,j,2)*s(Ydot,i,j,0);
            s(i_left,i,j,0) += 1./(2*dx) * ( pow(s(Y,i,j+1,0),2)*s(Y,i,j+1,2) + 0.5*g*pow(s(Y,i,j+1,2),2)
                                            -pow(s(Y,i,j-1,0),2)*s(Y,i,j-1,2) - 0.5*g*pow(s(Y,i,j-1,2),2))
                              + 1./(2*dy) * (-s(Y,i+1,j,0)*s(Y,i+1,j,1)*s(Y,i+1,j,2)
                                            +s(Y,i-1,j,0)*s(Y,i-1,j,1)*s(Y,i-1,j,2));
            
            s(i_left,i,j,1) = s(Y,i,j,2)*s(Ydot,i,j,1) + s(Y,i,j,1)*s(Ydot,i,j,2);
            s(i_left,i,j,1) += 1./(2*dy) * ( pow(s(Y,i-1,j,1),2)*s(Y,i-1,j,2) + 0.5*g*pow(s(Y,i-1,j,2),2)
                                            -pow(s(Y,i+1,j,1),2)*s(Y,i+1,j,2) - 0.5*g*pow(s(Y,i+1,j,2),2))
                              + 1./(2*dx) * (s(Y,i,j+1,0)*s(Y,i,j+1,1)*s(Y,i,j+1,2)
                                            -s(Y,i,j-1,0)*s(Y,i,j-1,1)*s(Y,i,j-1,2));
                    
            s(i_left,i,j,2) = s(Ydot,i,j,2) + 1./(2*dx)*(s(Y,i,j+1,0)*s(Y,i,j+1,2) - s(Y,i,j-1,0)*s(Y,i,j-1,2))
                              + 1./(2*dy)*(-s(Y,i+1,j,1)*s(Y,i+1,j,2) + s(Y,i-1,j,1)*s(Y,i-1,j,2));
        }
    }
    
    return i_left;
}

arma::vec form_RHS(
    double t, 
    arma::vec& Y,
    Structure2d& structure
){
    // return zeros like Y
    arma::vec rhs(Y.n_rows, arma::fill::zeros);
    return rhs;
}

arma::sp_mat form_Jacobian_Y(
    double t, 
    arma::vec& Y,
    arma::vec& Ydot,
    Structure2d& s
){
    double dx = s.m_dx;
    double dy = s.m_dy;

    arma::sp_mat jacobian(Y.n_rows, Y.n_rows);

    for(int j=0; j<s.m_grid_x; j++){
        for(int i=0; i<s.m_grid_y; i++){
            // form jacobian for Y 
            // if (i, j) is on boundary, then we set the jacobian to be identity
            if(i == 0 or i == s.m_grid_y-1 or j == 0 or j == s.m_grid_x-1){
                s(jacobian,i,j,0, i,j,0) += 1;
                s(jacobian,i,j,1, i,j,1) += 1;
                s(jacobian,i,j,2, i,j,2) += 1;
            }else{
                s(jacobian,i,j,0, i,j,0) += s(Ydot,i,j,2);
                s(jacobian,i,j,0, i,j,2) += s(Ydot,i,j,0); 
                s(jacobian,i,j,0, i,j+1,0) += 1./(2*dx) * (2*s(Y,i,j+1,0)*s(Y,i,j+1,2));
                s(jacobian,i,j,0, i,j+1,2) += 1./(2*dx) * (pow(s(Y,i,j+1,0),2) + g*s(Y,i,j+1,2));
                s(jacobian,i,j,0, i,j-1,0) -= 1./(2*dx) * (2*s(Y,i,j-1,0)*s(Y,i,j-1,2));
                s(jacobian,i,j,0, i,j-1,2) -= 1./(2*dx) * (pow(s(Y,i,j-1,0),2) + g*s(Y,i,j-1,2));
                s(jacobian,i,j,0, i+1,j,0) -= 1./(2*dy) * (s(Y,i+1,j,1)*s(Y,i+1,j,2));
                s(jacobian,i,j,0, i+1,j,1) -= 1./(2*dy) * (s(Y,i+1,j,0)*s(Y,i+1,j,2));
                s(jacobian,i,j,0, i+1,j,2) -= 1./(2*dy) * (s(Y,i+1,j,0)*s(Y,i+1,j,1));
                s(jacobian,i,j,0, i-1,j,0) += 1./(2*dy) * (s(Y,i-1,j,1)*s(Y,i-1,j,2));
                s(jacobian,i,j,0, i-1,j,1) += 1./(2*dy) * (s(Y,i-1,j,0)*s(Y,i-1,j,2));
                s(jacobian,i,j,0, i-1,j,2) += 1./(2*dy) * (s(Y,i-1,j,0)*s(Y,i-1,j,1));

                s(jacobian,i,j,1, i,j,1) += s(Ydot,i,j,2);
                s(jacobian,i,j,1, i,j,2) += s(Ydot,i,j,1);
                s(jacobian,i,j,1, i,j+1,0) += 1./(2*dx) * (s(Y,i,j+1,2)*s(Y,i,j+1,1));
                s(jacobian,i,j,1, i,j+1,1) += 1./(2*dx) * (s(Y,i,j+1,0)*s(Y,i,j+1,2));
                s(jacobian,i,j,1, i,j+1,2) += 1./(2*dx) * (s(Y,i,j+1,1)*s(Y,i,j+1,0));
                s(jacobian,i,j,1, i,j-1,0) -= 1./(2*dx) * (s(Y,i,j-1,2)*s(Y,i,j-1,1));
                s(jacobian,i,j,1, i,j-1,1) -= 1./(2*dx) * (s(Y,i,j-1,0)*s(Y,i,j-1,2));
                s(jacobian,i,j,1, i,j-1,2) -= 1./(2*dx) * (s(Y,i,j-1,1)*s(Y,i,j-1,0));
                s(jacobian,i,j,1, i+1,j,1) -= 1./(2*dy) * (2*s(Y,i+1,j,1)*s(Y,i+1,j,2));
                s(jacobian,i,j,1, i+1,j,2) -= 1./(2*dy) * (pow(s(Y,i+1,j,1),2) + g*s(Y,i+1,j,2));
                s(jacobian,i,j,1, i-1,j,1) += 1./(2*dy) * (2*s(Y,i-1,j,1)*s(Y,i-1,j,2));
                s(jacobian,i,j,1, i-1,j,2) += 1./(2*dy) * (pow(s(Y,i-1,j,1),2) + g*s(Y,i-1,j,2));
                
                s(jacobian,i,j,2, i,j+1,0) += 1./(2*dx) * s(Y,i,j+1,2);
                s(jacobian,i,j,2, i,j+1,2) += 1./(2*dx) * s(Y,i,j+1,0);
                s(jacobian,i,j,2, i,j-1,0) -= 1./(2*dx) * s(Y,i,j-1,2);
                s(jacobian,i,j,2, i,j-1,2) -= 1./(2*dx) * s(Y,i,j-1,0);
                s(jacobian,i,j,2, i+1,j,1) -= 1./(2*dy) * s(Y,i+1,j,2);
                s(jacobian,i,j,2, i+1,j,2) -= 1./(2*dy) * s(Y,i+1,j,1);
                s(jacobian,i,j,2, i-1,j,1) += 1./(2*dy) * s(Y,i-1,j,2);
                s(jacobian,i,j,2, i-1,j,2) += 1./(2*dy) * s(Y,i-1,j,1);
            }
        }
    }

    return jacobian;
}

arma::sp_mat form_Jacobian_Ydot(
    double t, 
    arma::vec& Y,
    arma::vec& Ydot,
    Structure2d& s
){
    double dx = s.m_dx;
    double dy = s.m_dy;

    arma::sp_mat jacobian(Y.n_rows, Y.n_rows);

    for(int j=0; j<s.m_grid_x; j++){
        for(int i=0; i<s.m_grid_y; i++){
            // if (i, j) is on boundary, then we set the jacobian to be identity
            if(i == 0 or i == s.m_grid_y-1 or j == 0 or j == s.m_grid_x-1){
                continue;
            }else{
                s(jacobian,i,j,0, i,j,0, s(Y,i,j,2));
                s(jacobian,i,j,0, i,j,2, s(Y,i,j,0));

                s(jacobian,i,j,1, i,j,1, s(Y,i,j,2));
                s(jacobian,i,j,1, i,j,2, s(Y,i,j,1));

                s(jacobian,i,j,2, i,j,2, 1);
            }
        }
    }

    return jacobian;
}

// function solver for shallow water equation, using rk2
arma::vec model(
    double t, 
    double dt,
    arma::vec& Y,
    Structure2d& structure,
    double tol=1e-6
){
    // tol *= structure.m_grid_x * structure.m_grid_y;
    // lambda function to calculate loss
    // using t, Y, Ydot, structure
    auto form_loss = [dt](double t, arma::vec& Y, arma::vec& Ydot, Structure2d& structure)->arma::vec{
        arma::vec i_left = form_Ifunction(t, Y, Ydot, structure);
        arma::vec i_right = form_RHS(t, Y, structure);
        return i_left - i_right;
    };

    arma::vec Y_next = Y;
    int try_count=0;

    while(true){
        arma::vec Ydot = (Y_next - Y) / dt;
        arma::vec Y_mid = (Y_next + Y) / 2;
        arma::vec loss_vec = form_loss(t+dt/2, Y_mid, Ydot, structure);
        double loss = arma::norm(loss_vec, 2);

        // std::cout<<Y_next<<loss_vec;
        std::cout<<"time: "<<t<<" time stepping "<<try_count++<<", loss is "<<loss<<std::endl;
        
        if(loss < tol){
            break;
        }

        arma::sp_mat jacobian1 = form_Jacobian_Y(t+dt/2, Y_mid, Ydot, structure);
        arma::sp_mat jacobian2 = form_Jacobian_Ydot(t+dt/2, Y_mid, Ydot, structure);
        arma::sp_mat jacobian = 0.5 * jacobian1 + 1./ dt * jacobian2;
        // std::cout<<jacobian<<std::endl;
        Y_next += arma::spsolve(jacobian, -loss_vec);

        // if(try_count > 100){
        //     throw std::runtime_error("time stepping failed");
        // }

        // we do line research here for y_next.
        // first write a lambda function to calculate loss with given step length
        // auto form_loss_with_step_length = [&t,dt,&Y,&Y_next,&jacobian,&structure,&form_loss](double step_length)->double{
        //     std::vector<arma::mat> Y_temp = Y_next;
        //     for(int i=0; i<Y_temp.size(); ++i){
        //         Y_temp[i] -= step_length * jacobian[i];
        //     }
        //     std::vector<arma::mat> Ydot = Y;
        //     for(int i=0; i<Ydot.size(); ++i){
        //         Ydot[i] = (Y_temp[i] - Y[i]) / delta_t;
        //     }

        //     double loss = form_loss(t, delta_t, Y, Ydot, structure);
        //     return loss;
        // };

        // // start with step length=1, test goldstein condition
        // // if not satisfied, let setp length *= rho
        // double step_length = 1;
        // double rho = 0.8;
        // double c = 0.1;
        // double dot_product = 0;
        // for(const auto& j: jacobian){
        //     dot_product += std::pow(arma::norm(j, "fro"), 2);
        // }
        // while(true){
        //     double loss_new = form_loss_with_step_length(step_length);
        //     if(loss_new <= loss - c * step_length * dot_product and
        //         loss_new >= loss - (1-c) * step_length * dot_product){
        //         break;
        //     }
        //     step_length *= rho;
        // }

        // // update Y_next
        // for(int i=0; i<Y_next.size(); ++i){
        //     Y_next[i] -= step_length * jacobian[i];
        // }
    }
    
    return Y_next;
}

double init_u(double x, double y){
    return 0;
}

double init_v(double x, double y){
    return 0;
}

double init_h(double x, double y){
    double distance = std::sqrt(x*x + y*y);
    if(distance < 1){
        return 2;
    }
    return 1;
}

void init(arma::vec &uvh, Structure2d &s){
    for(int j=0; j<s.m_grid_x; ++j){
        double x = s.m_left + j * s.m_dx;
        for(int i=0; i<s.m_grid_y; ++i){
            double y = s.m_high - i * s.m_dy;
            s(uvh,i,j,0) = init_u(x, y);
            s(uvh,i,j,1) = init_v(x, y);
            s(uvh,i,j,2) = init_h(x, y);
        }
    }
}

// write a function to test if all these functions work well
void test_shallow_water(){
    double t = 0;
    double delta_t = 1e-3;
    Structure2d structure(-2.5, 2.5, 513, -2.5, 2.5, 513, 3);

    arma::vec sol;
    structure.allocate_fields(sol);

    init(sol, structure);
    std::cout<<"t= "<<t<<std::endl;
    sol.save("./data/shallowwater/sol" + std::to_string(t) + ".csv", arma::raw_ascii);

    // we do this loop: save uvh, calculate uvh at next time, until t=0.2
    while(t < 0.2){
        sol = model(t, delta_t, sol, structure);
        t += delta_t;
        std::cout<<"t= "<<t<<std::endl;
        sol.save("./data/shallowwater/sol" + std::to_string(t) + ".csv", arma::raw_ascii);
    }
}

int main(){
    test_shallow_water();
    return 0;
}