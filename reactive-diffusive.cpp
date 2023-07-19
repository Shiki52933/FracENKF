#include "StochasticENKF.hpp"
#include "gEnKF.hpp"
#include <limits>
#include <cmath>
#include <assert.h>
#include <boost/program_options.hpp>

using namespace shiki;

double Du = 1;
double Dv = 1;
double phi = 0.1;
double k = 0.1;
double noise = 0.1;
double dt = 0.01;


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
        BoundaryCondition bc=BoundaryCondition::Periodic
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
        // assert(i >= -1 && i <= m_grid_y);
        // assert(j >= -1 && j <= m_grid_x);
        // assert(k >= 0 && k < m_unknowns);
        i = (i + m_grid_y) % m_grid_y;
        j = (j + m_grid_x) % m_grid_x;
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
        // assert(yi >= -1 && yi <= m_grid_y);
        // assert(yj >= -1 && yj <= m_grid_x);
        // assert(yk >= 0 && yk <= m_unknowns);
        yi = (yi + m_grid_y) % m_grid_y;
        yj = (yj + m_grid_x) % m_grid_x;
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
        // assert(yi >= -1 && yi <= m_grid_y);
        // assert(yj >= -1 && yj <= m_grid_x);
        // assert(yk >= -1 && yk <= m_unknowns);
        yi = (yi + m_grid_y) % m_grid_y;
        yj = (yj + m_grid_x) % m_grid_x;
        return jacobian(fk*m_grid_y*m_grid_x + fj*m_grid_y + fi, yk*m_grid_y*m_grid_x + yj*m_grid_y + yi);
        }
};

arma::vec form_Ifunction(
    double t, 
    arma::vec& Y,
    arma::vec& Ydot,
    Structure2d& s
){
    arma::vec I;
    s.allocate_fields(I);

    for(int i = 0; i < s.m_grid_y; i++){
        for(int j = 0; j < s.m_grid_x; j++){
            double u = s(Y, i, j, 0);
            double v = s(Y, i, j, 1);
            double udot = s(Ydot, i, j, 0);
            double vdot = s(Ydot, i, j, 1);
            
            // use 9-point to calculate second order derivative
            double lap_u = (s(Y,i-1,j-1,0) + 4*s(Y,i-1,j,0) + s(Y,i-1,j+1,0)
                            + 4*s(Y,i,j-1,0) - 20*s(Y,i,j,0) + 4*s(Y,i,j+1,0)
                            + s(Y,i+1,j-1,0) + 4*s(Y,i+1,j,0) + s(Y,i+1,j+1,0)) 
                            / (6*s.m_dx*s.m_dx);
            double lap_v = (s(Y,i-1,j-1,1) + 4*s(Y,i-1,j,1) + s(Y,i-1,j+1,1)
                            + 4*s(Y,i,j-1,1) - 20*s(Y,i,j,1) + 4*s(Y,i,j+1,1)
                            + s(Y,i+1,j-1,1) + 4*s(Y,i+1,j,1) + s(Y,i+1,j+1,1)) 
                            / (6*s.m_dx*s.m_dx);
            
            s(I, i, j, 0) = udot - Du*lap_u + u*v*v - phi*(1 - u);
            s(I, i, j, 1) = vdot - Dv*lap_v - u*v*v + (phi+k)*v;
        }
    }
    return I;
}

arma::sp_mat form_Jacobian_Y(
    double t, 
    arma::vec& Y,
    arma::vec& Ydot,
    Structure2d& s
){
    arma::sp_mat J(Y.n_rows, Y.n_rows);

    for(int j=0; j<s.m_grid_y; ++j){
        for(int i=0; i<s.m_grid_x; ++i){
            double u = s(Y, i, j, 0);
            double v = s(Y, i, j, 1);
            double udot = s(Ydot, i, j, 0);
            double vdot = s(Ydot, i, j, 1);
            
            s(J, i, j, 0, i-1, j-1, 0) = -Du/(6*s.m_dy*s.m_dy);
            s(J, i, j, 0, i-1, j, 0) = -4*Du/(6*s.m_dy*s.m_dy);
            s(J, i, j, 0, i-1, j+1, 0) = -Du/(6*s.m_dy*s.m_dy);
            s(J, i, j, 0, i, j-1, 0) = -4*Du/(6*s.m_dx*s.m_dx);
            s(J, i, j, 0, i, j, 0) = phi + 20*Du/(6*s.m_dx*s.m_dx) + v*v;
            s(J, i, j, 0, i, j+1, 0) = -4*Du/(6*s.m_dx*s.m_dx);
            s(J, i, j, 0, i+1, j-1, 0) = -Du/(6*s.m_dy*s.m_dy);
            s(J, i, j, 0, i+1, j, 0) = -4*Du/(6*s.m_dy*s.m_dy);
            s(J, i, j, 0, i+1, j+1, 0) = -Du/(6*s.m_dy*s.m_dy);
            s(J, i, j, 0, i, j, 1) = 2*u*v;

            s(J, i, j, 1, i-1, j-1, 1) = -Dv/(6*s.m_dy*s.m_dy);
            s(J, i, j, 1, i-1, j, 1) = -4*Dv/(6*s.m_dy*s.m_dy);
            s(J, i, j, 1, i-1, j+1, 1) = -Dv/(6*s.m_dy*s.m_dy);
            s(J, i, j, 1, i, j-1, 1) = -4*Dv/(6*s.m_dx*s.m_dx);
            s(J, i, j, 1, i, j, 1) = phi + k + 20*Dv/(6*s.m_dx*s.m_dx) - 2*u*v;
            s(J, i, j, 1, i, j+1, 1) = -4*Dv/(6*s.m_dx*s.m_dx);
            s(J, i, j, 1, i+1, j-1, 1) = -Dv/(6*s.m_dy*s.m_dy);
            s(J, i, j, 1, i+1, j, 1) = -4*Dv/(6*s.m_dy*s.m_dy);
            s(J, i, j, 1, i+1, j+1, 1) = -Dv/(6*s.m_dy*s.m_dy);
            s(J, i, j, 1, i, j, 0) = -v*v;

        }
    }
    return J;
}

arma::sp_mat form_Jacobian_Ydot(
    double t, 
    arma::vec& Y,
    arma::vec& Ydot,
    Structure2d& s
){
    arma::sp_mat J(Y.n_rows, Y.n_rows);

    for(int j=0; j<s.m_grid_y; ++j){
        for(int i=0; i<s.m_grid_x; ++i){
            s(J, i, j, 0, i, j, 0) = 1;
            s(J, i, j, 1, i, j, 1) = 1;
        }
    }
    return J;
}

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
        // arma::vec i_right = form_RHS(t, Y, structure);
        return i_left;
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
    }
    return Y_next;
}

void init(arma::vec &uv, Structure2d &s, double L){
    uv = std::max(0., noise) * arma::randu(uv.n_rows);

    double ledge = (L - 0.5) / 2.;
    double redge = L - ledge;

    for(int j=0; j<s.m_grid_x; ++j){
        double x = s.m_left + j * s.m_dx;
        for(int i=0; i<s.m_grid_y; ++i){
            double y = s.m_high - i * s.m_dy;

            if(x >= ledge and x <= redge and y >= ledge and y <= redge){
                double sx = std::sin(4*M_PI*x);
                double sy = std::sin(4*M_PI*y);
                s(uv, i, j, 1) += 0.5 * sx * sx * sy * sy;
            }
            s(uv, i, j, 0) += 1 - 2 * s(uv, i, j, 1);
        }
    }
}

void test(int grid_x, int grid_y, double L, double tmax){
    Structure2d s(0, L, grid_x, 0, L, grid_y, 2);
    arma::vec sol;
    s.allocate_fields(sol);

    init(sol, s, L);
    sol.save("./data/re-di/sol" + std::to_string(0) + ".bin", arma::raw_binary);

    double t = 0;
    while(t < tmax){
        sol = model(t, dt, sol, s);
        t += dt;
        sol.save("./data/re-di/sol" + std::to_string(t) + ".bin", arma::raw_binary);
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
        ("tmax", po::value<double>()->default_value(1000), "max time")
        ("dt", po::value<double>(&dt)->default_value(5), "dt")
        ("Du", po::value<double>(&Du)->default_value(8e-5), "Du")
        ("Dv", po::value<double>(&Dv)->default_value(4e-5), "Dv")
        ("phi", po::value<double>(&phi)->default_value(0.024), "phi")
        ("k", po::value<double>(&k)->default_value(0.06), "k")
        ("L", po::value<double>()->default_value(2.5), "gravity")
        ("noise", po::value<double>(&noise)->default_value(0), "noise");
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    for(auto& t:vm){
        std::cout<<t.first<<" ";
        auto& value = t.second.value();
        if(auto v = boost::any_cast<int>(&value)){
            std::cout<<*v<<std::endl;
        }else if(auto v = boost::any_cast<double>(&value)){
            std::cout<<*v<<std::endl;
        }else{
            std::cout<<std::endl;
        }
    }

    test(vm["grid_x"].as<int>(), vm["grid_y"].as<int>(), vm["L"].as<double>(), vm["tmax"].as<double>());

    return 0;
}