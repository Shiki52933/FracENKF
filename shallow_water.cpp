#include "StochasticENKF.hpp"
#include "gEnKF.hpp"
#include <limits>
#include <cmath>
#include <assert.h>
#include <boost/program_options.hpp>

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

double g = 9;
double ave_h = 1;

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
        // assert(i >= 0 && i < m_grid_y);
        // assert(j >= 0 && j < m_grid_x);
        // assert(k >= 0 && k < m_unknowns);
        return fields(k*m_grid_y*m_grid_x + j*m_grid_y + i);
    }

    // overload () operator to set jacobian
    inline void operator()(
        arma::sp_mat& jacobian,
        int fi, int fj, int fk,
        int yi, int yj, int yk,
        double value
        ){
        // assert(fi >= 0 && fi < m_grid_y);
        // assert(fj >= 0 && fj < m_grid_x);
        // assert(fk >= 0 && fk < m_unknowns);
        // assert(yi >= 0 && yi < m_grid_y);
        // assert(yj >= 0 && yj < m_grid_x);
        // assert(yk >= 0 && yk < m_unknowns);
        jacobian(fk*m_grid_y*m_grid_x + fj*m_grid_y + fi, yk*m_grid_y*m_grid_x + yj*m_grid_y + yi) = value;
        }
    
    inline auto operator()(
        arma::sp_mat& jacobian,
        int fi, int fj, int fk,
        int yi, int yj, int yk
        ){
        // assert(fi >= 0 && fi < m_grid_y);
        // assert(fj >= 0 && fj < m_grid_x);
        // assert(fk >= 0 && fk < m_unknowns);
        // assert(yi >= 0 && yi < m_grid_y);
        // assert(yj >= 0 && yj < m_grid_x);
        // assert(yk >= 0 && yk < m_unknowns);
        return jacobian(fk*m_grid_y*m_grid_x + fj*m_grid_y + fi, yk*m_grid_y*m_grid_x + yj*m_grid_y + yi);
        }

    double average_h(arma::vec& fields){
        double sum = 0;
        for(int i = 0; i < m_grid_y; i++){
            for(int j = 0; j < m_grid_x; j++){
                sum += (*this)(fields, i, j, 2);
            }
        }
        return sum / (m_grid_x * m_grid_y);
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

    // write a lambda function to access normal fields or ghost fields
    auto access = [&s](arma::vec& f, int i, int j, int k){
        double flip = 1;
        if(i < 0){
            i = 1;
            if(k == 1){
                flip = -1;
            }
        }else if(i >= s.m_grid_y){
            i = s.m_grid_y - 2;
            if(k == 1){
                flip = -1;
            }
        }else if(j < 0){
            j = 1;
            if(k == 0){
                flip = -1;
            }
        }else if(j >= s.m_grid_x){
            j = s.m_grid_x - 2;
            if(k == 0){
                flip = -1;
            }
        }

        return flip * s(f, i, j, k);
    };

    for(int j=0; j<s.m_grid_x; j++){
        for(int i=0; i<s.m_grid_y; i++){
            s(i_left,i,j,0) = access(Y,i,j,0)*access(Ydot,i,j,2) + access(Y,i,j,2)*access(Ydot,i,j,0);
            s(i_left,i,j,0) += 1./(2*dx) * ( pow(access(Y,i,j+1,0),2)*access(Y,i,j+1,2) + 0.5*g*pow(access(Y,i,j+1,2),2)
                                            -pow(access(Y,i,j-1,0),2)*access(Y,i,j-1,2) - 0.5*g*pow(access(Y,i,j-1,2),2))
                              + 1./(2*dy) * (-access(Y,i+1,j,0)*access(Y,i+1,j,1)*access(Y,i+1,j,2)
                                            +access(Y,i-1,j,0)*access(Y,i-1,j,1)*access(Y,i-1,j,2));
            
            s(i_left,i,j,1) = access(Y,i,j,2)*access(Ydot,i,j,1) + access(Y,i,j,1)*access(Ydot,i,j,2);
            s(i_left,i,j,1) += 1./(2*dy) * ( pow(access(Y,i-1,j,1),2)*access(Y,i-1,j,2) + 0.5*g*pow(access(Y,i-1,j,2),2)
                                            -pow(access(Y,i+1,j,1),2)*access(Y,i+1,j,2) - 0.5*g*pow(access(Y,i+1,j,2),2))
                              + 1./(2*dx) * (access(Y,i,j+1,0)*access(Y,i,j+1,1)*access(Y,i,j+1,2)
                                            -access(Y,i,j-1,0)*access(Y,i,j-1,1)*access(Y,i,j-1,2));
                    
            s(i_left,i,j,2) = access(Ydot,i,j,2) 
                              + 1./(2*dx)*(access(Y,i,j+1,0)*access(Y,i,j+1,2) - access(Y,i,j-1,0)*access(Y,i,j-1,2))
                              + 1./(2*dy)*(-access(Y,i+1,j,1)*access(Y,i+1,j,2) + access(Y,i-1,j,1)*access(Y,i-1,j,2));      
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

    auto access = [&s](arma::vec& f, int i, int j, int k){
        double flip = 1;
        if(i < 0){
            i = 1;
            if(k == 1){
                flip = -1;
            }
        }else if(i >= s.m_grid_y){
            i = s.m_grid_y - 2;
            if(k == 1){
                flip = -1;
            }
        }else if(j < 0){
            j = 1;
            if(k == 0){
                flip = -1;
            }
        }else if(j >= s.m_grid_x){
            j = s.m_grid_x - 2;
            if(k == 0){
                flip = -1;
            }
        }

        return flip * s(f, i, j, k);
    };

    auto add = [&s](
        arma::sp_mat& jacobian, 
        int fi, int fj, int fk, 
        int yi, int yj, int yk, 
        double val){
        if(yi < 0){
            yi = 1;
            if(yk==1)
                val *= -1;
        }else if(yi>=s.m_grid_y){
            yi = s.m_grid_y-2;
            if(yk==1)
                val *= -1;
        }else if(yj < 0){
            yj = 1;
            if(yk==0)
                val *= -1;
        }else if(yj>=s.m_grid_x){
            yj = s.m_grid_x-2;
            if(yk==0)
                val *= -1;
        }
        s(jacobian, fi, fj, fk, yi, yj, yk) += val;
    };

    for(int j=0; j<s.m_grid_x; j++){
        for(int i=0; i<s.m_grid_y; i++){
                add(jacobian,i,j,0, i,j,0, access(Ydot,i,j,2));
                add(jacobian,i,j,0, i,j,2, access(Ydot,i,j,0)); 
                add(jacobian,i,j,0, i,j+1,0, 1./(2*dx) * (2*access(Y,i,j+1,0)*access(Y,i,j+1,2)));
                add(jacobian,i,j,0, i,j+1,2, 1./(2*dx) * (pow(access(Y,i,j+1,0),2) + g*access(Y,i,j+1,2)));
                add(jacobian,i,j,0, i,j-1,0, -1./(2*dx) * (2*access(Y,i,j-1,0)*access(Y,i,j-1,2)));
                add(jacobian,i,j,0, i,j-1,2, -1./(2*dx) * (pow(access(Y,i,j-1,0),2) + g*access(Y,i,j-1,2)));
                add(jacobian,i,j,0, i+1,j,0, -1./(2*dy) * (access(Y,i+1,j,1)*access(Y,i+1,j,2)));
                add(jacobian,i,j,0, i+1,j,1, -1./(2*dy) * (access(Y,i+1,j,0)*access(Y,i+1,j,2)));
                add(jacobian,i,j,0, i+1,j,2, -1./(2*dy) * (access(Y,i+1,j,0)*access(Y,i+1,j,1)));
                add(jacobian,i,j,0, i-1,j,0, +1./(2*dy) * (access(Y,i-1,j,1)*access(Y,i-1,j,2)));
                add(jacobian,i,j,0, i-1,j,1, +1./(2*dy) * (access(Y,i-1,j,0)*access(Y,i-1,j,2)));
                add(jacobian,i,j,0, i-1,j,2, +1./(2*dy) * (access(Y,i-1,j,0)*access(Y,i-1,j,1)));

                add(jacobian,i,j,1, i,j,1, access(Ydot,i,j,2));
                add(jacobian,i,j,1, i,j,2, access(Ydot,i,j,1));
                add(jacobian,i,j,1, i,j+1,0, 1./(2*dx) * (access(Y,i,j+1,2)*access(Y,i,j+1,1)));
                add(jacobian,i,j,1, i,j+1,1, 1./(2*dx) * (access(Y,i,j+1,0)*access(Y,i,j+1,2)));
                add(jacobian,i,j,1, i,j+1,2, 1./(2*dx) * (access(Y,i,j+1,1)*access(Y,i,j+1,0)));
                add(jacobian,i,j,1, i,j-1,0, -1./(2*dx) * (access(Y,i,j-1,2)*access(Y,i,j-1,1)));
                add(jacobian,i,j,1, i,j-1,1, -1./(2*dx) * (access(Y,i,j-1,0)*access(Y,i,j-1,2)));
                add(jacobian,i,j,1, i,j-1,2, -1./(2*dx) * (access(Y,i,j-1,1)*access(Y,i,j-1,0)));
                add(jacobian,i,j,1, i+1,j,1, -1./(2*dy) * (2*access(Y,i+1,j,1)*access(Y,i+1,j,2)));
                add(jacobian,i,j,1, i+1,j,2, -1./(2*dy) * (pow(access(Y,i+1,j,1),2) + g*access(Y,i+1,j,2)));
                add(jacobian,i,j,1, i-1,j,1, 1./(2*dy) * (2*access(Y,i-1,j,1)*access(Y,i-1,j,2)));
                add(jacobian,i,j,1, i-1,j,2, 1./(2*dy) * (pow(access(Y,i-1,j,1),2) + g*access(Y,i-1,j,2)));
                
                add(jacobian,i,j,2, i,j+1,0, 1./(2*dx) * access(Y,i,j+1,2));
                add(jacobian,i,j,2, i,j+1,2, 1./(2*dx) * access(Y,i,j+1,0));
                add(jacobian,i,j,2, i,j-1,0, -1./(2*dx) * access(Y,i,j-1,2));
                add(jacobian,i,j,2, i,j-1,2, -1./(2*dx) * access(Y,i,j-1,0));
                add(jacobian,i,j,2, i+1,j,1, -1./(2*dy) * access(Y,i+1,j,2));
                add(jacobian,i,j,2, i+1,j,2, -1./(2*dy) * access(Y,i+1,j,1));
                add(jacobian,i,j,2, i-1,j,1, 1./(2*dy) * access(Y,i-1,j,2));
                add(jacobian,i,j,2, i-1,j,2, 1./(2*dy) * access(Y,i-1,j,1));
            
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
                s(jacobian,i,j,0, i,j,0, s(Y,i,j,2));
                s(jacobian,i,j,0, i,j,2, s(Y,i,j,0));

                s(jacobian,i,j,1, i,j,1, s(Y,i,j,2));
                s(jacobian,i,j,1, i,j,2, s(Y,i,j,1));

                s(jacobian,i,j,2, i,j,2, 1);     
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
        // std::cout<<(arma::mat)jacobian<<std::endl;
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
void test_shallow_water(int gx, int gy, double tmax){
    double t = 0;
    double L=5;
    Structure2d structure(-L/2, L/2, gx, -L/2, L/2, gy, 3);
    double delta_t = 0.001;

    std::cout<<"grid_x: "<<structure.m_grid_x<<std::endl;
    std::cout<<"grid_y: "<<structure.m_grid_y<<std::endl;
    std::cout<<"dt: "<<delta_t<<" tmax: "<<tmax<<std::endl;

    arma::vec sol;
    structure.allocate_fields(sol);

    init(sol, structure);
    // update ave_h
    ave_h = structure.average_h(sol);
    std::cout<<"t= "<<t<<"average h= "<<ave_h<<std::endl;
    sol.save("./data/shallowwater/sol" + std::to_string(t) + ".bin", arma::raw_binary);

    // we do this loop: save uvh, calculate uvh at next time, until t=0.2
    while(t < tmax){
        sol = model(t, delta_t, sol, structure);
        t += delta_t;

        ave_h = structure.average_h(sol);
        std::cout<<"t= "<<t<<" average h= "<<ave_h<<std::endl;
        sol.save("./data/shallowwater/sol" + std::to_string(t) + ".bin", arma::raw_binary);
    }
}

int main(int argc, char** argv){
    // use boost to parse command line
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("grid_x", po::value<int>()->default_value(100), "grid number in x direction")
        ("grid_y", po::value<int>()->default_value(100), "grid number in y direction")
        ("tmax", po::value<double>()->default_value(0.2), "max time")
        ("g", po::value<double>(&g)->default_value(9.8), "gravity");
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    test_shallow_water(vm["grid_x"].as<int>(), vm["grid_y"].as<int>(), vm["tmax"].as<double>());

    return 0;
}