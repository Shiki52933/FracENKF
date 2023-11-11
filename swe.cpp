#include "StochasticENKF.hpp"
#include "gEnKF.hpp"
#include <limits>
#include <cmath>
#include <assert.h>
#include <boost/program_options.hpp>

using namespace shiki;


enum class BoundaryCondition{
    Periodic,
    Reflective,
    Open
};

class SweSetting{
public:
    double H;
    double g;
    double kappa;
    arma::vec f; 
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

    SweSetting m_setting;

public:
    Structure2d(
        double left, 
        double right, 
        int grid_x, 
        double low, 
        double high, 
        int grid_y, 
        int unknowns,
        BoundaryCondition bc=BoundaryCondition::Reflective
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
    
    inline double& operator()(arma::vec &f, int i, int j){
        // assert(i >= 0 && i < m_grid_y);
        // assert(j >= 0 && j < m_grid_x);
        return f(j*m_grid_y + i);
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
        return fields(j*m_grid_y*m_unknowns + i*m_unknowns + k);
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
        jacobian(fj*m_grid_y*m_unknowns + fi*m_unknowns + fk, yj*m_grid_y*m_unknowns + yi*m_unknowns + yk) = value;
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
        return jacobian(fj*m_grid_y*m_unknowns + fi*m_unknowns + fk, yj*m_grid_y*m_unknowns + yi*m_unknowns + yk);
        }

    double average_h(arma::vec& fields){
        double sum = 0;
        for(int j = 0; j < m_grid_x; j++){
            for(int i = 0; i < m_grid_y; i++){
                sum += (*this)(fields, i, j, 2);
            }
        }
        return sum / (m_grid_x * m_grid_y);
    }

    inline int index2int(int i, int j, int k){
        return j*m_grid_y*m_unknowns + i*m_unknowns + k;
    }
};

/// @brief ifucntion for swe in nonconservative form
/// @brief for simplicity, pde is 
/// @brief du/dt - fv = -g*d(eta)/dx - kappa*u,
/// @brief dv/dt + fu = -g*d(eta)/dy - kappa*v,
/// @brief d(eta)/dt + d((eta + H)*u)/dx + d((eta + H)*v)/dy = 0.
/// @brief no plan for complete form recently
/// @param t 
/// @param Y 
/// @param Ydot 
/// @param s 
/// @return 
arma::vec form_Ifunction(
    double t, 
    arma::vec& Y,
    arma::vec& Ydot,
    Structure2d& s
){
    double dx = s.m_dx;
    double dy = s.m_dy;
    arma::vec& f = s.m_setting.f;
    double g = s.m_setting.g;
    double H = s.m_setting.H;
    double kappa = s.m_setting.kappa;

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
            double u = access(Y, i, j, 0);
            double v = access(Y, i, j, 1);
            double eta = access(Y, i, j, 2);
            double udot = access(Ydot, i, j, 0);
            double vdot = access(Ydot, i, j, 1);
            double etadot = access(Ydot, i, j, 2);

            double deta_dx = (access(Y, i, j+1, 2) - access(Y, i, j-1, 2)) / (2 * dx);
            double deta_dy = (access(Y, i-1, j, 2) - access(Y, i+1, j, 2)) / (2 * dy);

            double detaHu_dx = 1./(2*dx) * ((access(Y,i,j+1,2) + s.m_setting.H)*access(Y,i,j+1,0) 
                                        - (access(Y,i,j-1,2) + s.m_setting.H)*access(Y,i,j-1,0));
            double detaHv_dy = 1./(2*dy) * ((access(Y,i-1,j,2) + s.m_setting.H)*access(Y,i-1,j,1)
                                        - (access(Y,i+1,j,2) + s.m_setting.H)*access(Y,i+1,j,1));
            
            s(i_left,i,j,0) = udot - s(f,i,j)*v + g * deta_dx + kappa * u;
            s(i_left,i,j,1) = vdot + s(f,i,j)*u + g * deta_dy + kappa * v;
            s(i_left,i,j,2) = etadot + detaHu_dx + detaHv_dy;
        }
    }
    
    return i_left;
}

arma::sp_mat form_Jacobian_Y(
    double t, 
    arma::vec& Y,
    arma::vec& Ydot,
    Structure2d& s
){
    double dx = s.m_dx;
    double dy = s.m_dy;
    arma::vec& f = s.m_setting.f;
    double g = s.m_setting.g;
    double H = s.m_setting.H;
    double kappa = s.m_setting.kappa;

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
            add(jacobian, i,j,0, i,j,0, kappa);
            add(jacobian, i,j,0, i,j,1, -s(f,i,j)); 
            add(jacobian, i,j,0, i,j+1,2, g/(2*dx));
            add(jacobian, i,j,0, i,j-1,2, -g/(2*dx));

            add(jacobian, i,j,1, i,j,1, kappa);
            add(jacobian, i,j,1, i,j,0, s(f,i,j));
            add(jacobian, i,j,1, i+1,j,2, -g/(2*dy));
            add(jacobian, i,j,1, i-1,j,2, g/(2*dy));

            add(jacobian, i,j,2, i,j+1,0, 1./(2*dx)*(access(Y,i,j+1,2) + H));
            add(jacobian, i,j,2, i,j+1,2, 1./(2*dx)*access(Y,i,j+1,0));
            add(jacobian, i,j,2, i,j-1,0, -1./(2*dx)*(access(Y,i,j-1,2) + H));
            add(jacobian, i,j,2, i,j-1,2, -1./(2*dx)*access(Y,i,j-1,0));
            add(jacobian, i,j,2, i-1,j,1, 1./(2*dy)*(access(Y,i-1,j,2) + H));
            add(jacobian, i,j,2, i-1,j,2, 1./(2*dy)*access(Y,i-1,j,1));
            add(jacobian, i,j,2, i+1,j,1, -1./(2*dy)*(access(Y,i+1,j,2) + H));
            add(jacobian, i,j,2, i+1,j,2, -1./(2*dy)*access(Y,i+1,j,1));
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
            s(jacobian, i,j,0, i,j,0) += 1;
            s(jacobian, i,j,1, i,j,1) += 1;
            s(jacobian, i,j,2, i,j,2) += 1;     
        }
    }

    return jacobian;
}

// function solver for shallow water equation, using rk2
arma::vec model(
    double t, 
    double dt,
    const arma::vec& Y,
    Structure2d& structure,
    double tol=1e-6
){
    // tol *= structure.m_grid_x * structure.m_grid_y;
    // lambda function to calculate loss
    // using t, Y, Ydot, structure
    auto form_loss = [dt](double t, arma::vec& Y, arma::vec& Ydot, Structure2d& structure)->arma::vec{
        arma::vec i_left = form_Ifunction(t, Y, Ydot, structure);
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
        // std::cout<<"time: "<<t<<" time stepping "<<try_count++<<", loss is "<<loss<<std::endl;
        
        if(loss < tol){
            break;
        }

        arma::sp_mat jacobian1 = form_Jacobian_Y(t+dt/2, Y_mid, Ydot, structure);
        arma::sp_mat jacobian2 = form_Jacobian_Ydot(t+dt/2, Y_mid, Ydot, structure);
        arma::sp_mat jacobian = 0.5 * jacobian1 + 1./ dt * jacobian2;
        // std::cout<<jacobian<<std::endl;
        // std::cout<<(arma::mat)jacobian<<std::endl;
        Y_next += shiki::Krylov(jacobian, -loss_vec);
    }
    
    return Y_next;
}

double init_u(double x, double y){
    return 0;
}

double init_v(double x, double y){
    return 0;
}

// double init_h(double x, double y){
//     using std::exp;
//     using std::pow;
//     double cx = 1e6/2.7, cy = 1e6/4.;
//     double sx = 5e4, sy = 5e4;
//     return exp(
//         -(
//             pow(x-cx, 2) / (2 * pow(sx, 2)) + pow(y-cy, 2) / (2 * pow(sy, 2))
//         )
//     );
// }

void init(arma::vec &uvh, Structure2d &s, bool random=false, double init_error=0.03){
    double cx = 1e6/2.7, cy = 1e6/4.;
    double sx = 5e4, sy = 5e4;
    if(random){
        cx *= 1 + arma::randn() * init_error;
        cy *= 1 + arma::randn() * init_error;
        sx *= 1 + arma::randn() * init_error;
        sy *= 1 + arma::randn() * init_error;
    }
    auto init_h = [&cx, &cy, &sx, &sy](double x, double y){
        return std::exp(
            -(
            pow(x-cx, 2) / (2 * pow(sx, 2)) + pow(y-cy, 2) / (2 * pow(sy, 2))
            )  
        );
    };

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

void init_setting(Structure2d &s){
    s.m_setting.g = 9.8;
    s.m_setting.H = 100;
    s.m_setting.kappa = 1./(5.*24*3600);

    s.m_setting.f.set_size(s.m_grid_x*s.m_grid_y);
    for(int j=0; j<s.m_grid_x; ++j){
        double x = s.m_left + j * s.m_dx;
        for(int i=0; i<s.m_grid_y; ++i){
            double y = s.m_high - i * s.m_dy;
            s(s.m_setting.f,i,j) = 1e-4 + 2e-11 * y;
        }
    }
}

// write a function to test if all these functions work well
void test_shallow_water(int gx, int gy, int iter_times=5000){
    double t = 0;
    double L = 1e6;
    Structure2d structure(-L/2, L/2, gx, -L/2, L/2, gy, 3);
    init_setting(structure); 

    double delta_t = 0.1 * L / std::max(gx-1, gy-1) / 
                    std::sqrt(structure.m_setting.g * structure.m_setting.H);

    std::cout<<"grid_x: "<<structure.m_grid_x<<std::endl;
    std::cout<<"grid_y: "<<structure.m_grid_y<<std::endl;
    std::cout<<"dt: "<<delta_t<<" tmax: "<<(iter_times*delta_t)<<std::endl;

    arma::vec sol;
    structure.allocate_fields(sol);
    init(sol, structure);

    // update ave_h
    double ave_h = structure.average_h(sol);
    std::cout<<"t= "<<t<<"average h= "<<ave_h<<std::endl;
    sol.save("./data/swe/sol" + std::to_string(t) + ".bin", arma::raw_binary);

    // we do this loop: save uvh, calculate uvh at next time, until t=0.2
    while(t < iter_times * delta_t){
        sol = model(t, delta_t, sol, structure);
        t += delta_t;

        ave_h = structure.average_h(sol);
        std::cout<<"t= "<<t<<" average h= "<<ave_h<<std::endl;
        sol.save("./data/swe/sol" + std::to_string(t) + ".bin", arma::raw_binary);
    }
}

class SurrogateModel{
    using Model = arma::vec(
        double, 
        double,
        const arma::vec&,
        Structure2d&,
        double
        );
public:
    Structure2d& m_structure;
    Model *m_model;

    int m_iter = 0;
    arma::sp_mat m_H;
    std::vector<arma::vec> m_ob_results;


    SurrogateModel(
        Structure2d& structure, 
        Model f, 
        int num_ob_pos)
        : m_structure(structure), m_model(f){
        m_H.set_size(
            num_ob_pos, 
            structure.m_grid_x * structure.m_grid_y * structure.m_unknowns
            );
    }

    void add_ob_pos(double x, double y, int k){
        int i = std::round((m_structure.m_high - y) / m_structure.m_dy);
        int j = std::round((x - m_structure.m_left) / m_structure.m_dx);
        m_H(m_iter++, m_structure.index2int(i, j, k)) = 1;
    }

    void gen_numerical_solution(
        arma::vec& sol, 
        double t0, 
        double delta_t,
        int iter_times,
        double ob_noise){     
        sol.save("./data/swe/sol" + std::to_string(t0) + ".bin", arma::raw_binary);

        // loop: ob, model update
        for(int i=0; i<iter_times; ++i){
            // ob
            arma::vec ob = m_H * sol;
            ob += arma::randn(ob.n_elem) * ob_noise;
            m_ob_results.push_back(ob);
            // update sol
            sol = m_model(t0, delta_t, sol, m_structure, 1e-6);
            t0 += delta_t;
            sol.save("./data/swe/sol" + std::to_string(t0) + ".bin", arma::raw_binary);
        }
    }

};

void test_da(int gx, int gy, int iter_times, int en_size, int ob_num, double init_error){
    double t = 0;
    double L = 1e6;
    Structure2d structure(-L/2, L/2, gx, -L/2, L/2, gy, 3);
    init_setting(structure); 

    double delta_t = 0.1 * L / std::max(gx-1, gy-1) / 
                    std::sqrt(structure.m_setting.g * structure.m_setting.H);

    {
    std::cout<<"grid_x: "<<structure.m_grid_x<<std::endl;
    std::cout<<"grid_y: "<<structure.m_grid_y<<std::endl;
    std::cout<<"dt: "<<delta_t<<" tmax: "<<(iter_times*delta_t)<<std::endl;
    }

    // init sol
    arma::vec sol;
    structure.allocate_fields(sol);
    init(sol, structure);

    // set ob pos
    SurrogateModel s_model(structure, model, ob_num*ob_num);
    // we observe at ob_num*ob_num grids, boundary is also observed 
    double ob_dx = L / (ob_num - 1);
    double ob_dy = L / (ob_num - 1);
    for(int i=0; i<ob_num; ++i){
        for(int j=0; j<ob_num; ++j){
            double x = structure.m_left + j * ob_dx;
            double y = structure.m_high - i * ob_dy;
            s_model.add_ob_pos(x, y, 2);
        }
    }

    // generate numerical solution
    double ob_noise = 0.1;
    auto ob_error_ptr = std::make_shared<arma::mat>();
    *ob_error_ptr = arma::eye<arma::mat>(ob_num*ob_num, ob_num*ob_num) * ob_noise * ob_noise;
    s_model.gen_numerical_solution(sol, t, delta_t, iter_times, ob_noise);
    std::cout<<"numerical solution generated"<<std::endl;

    // prepare for use of pde_group_enkf in gEnKF.hpp
    // double init_error = 0.03;
    arma::mat ensemble(sol.n_elem, en_size, arma::fill::none);
    for(int i=0; i<en_size; ++i){
        init(sol, structure, true, init_error);
        ensemble.col(i) = sol;
    }
    ensemble.save("./data/swe_init.bin", arma::raw_binary);
    arma::sp_mat init_var = arma::eye<arma::sp_mat>(sol.n_elem, sol.n_elem) * init_error * init_error;
    std::vector<arma::sp_mat> vars = {init_var};
    std::cout<<"ensemble generated"<<std::endl;

    errors ob_error;
    for(int i=0; i<iter_times; ++i){
        ob_error.add(ob_error_ptr);
    }

    std::string output_dir = "./data/swe/";
    std::vector<double> max_error, rel_error;
    std::cout<<"start pde_group_Enkf"<<std::endl;

    pde_group_Enkf(
        iter_times, t, delta_t,
        ensemble, vars, 
        s_model.m_ob_results, s_model.m_H, ob_error,  
        model, structure, output_dir, 
        max_error, rel_error
        );

    // save max_error and rel_error
    arma::vec max_error_vec(max_error);
    arma::vec rel_error_vec(rel_error);
    max_error_vec.save("./data/max_error.bin", arma::raw_binary);
    rel_error_vec.save("./data/rel_error.bin", arma::raw_binary);
}

int main(int argc, char** argv){
    // use boost to parse command line
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("grid_x", po::value<int>()->default_value(150), "grid number in x direction")
        ("grid_y", po::value<int>()->default_value(150), "grid number in y direction")
        ("times", po::value<int>()->default_value(2000), "iter times")
        ("en-size", po::value<int>()->default_value(20), "ensemble size")
        ("ob-num", po::value<int>()->default_value(11), "ob number per direction")
        ("init-error", po::value<double>()->default_value(0.03), "init error");
        
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // print params got from command line
    std::cout<<"grid_x: "<<vm["grid_x"].as<int>()<<std::endl;
    std::cout<<"grid_y: "<<vm["grid_y"].as<int>()<<std::endl;
    std::cout<<"times: "<<vm["times"].as<int>()<<std::endl;
    std::cout<<"en-size: "<<vm["en-size"].as<int>()<<std::endl;
    std::cout<<"ob-num: "<<vm["ob-num"].as<int>()<<std::endl;
    std::cout<<"init-error: "<<vm["init-error"].as<double>()<<std::endl;
    

    test_da(
        vm["grid_x"].as<int>(), 
        vm["grid_y"].as<int>(), 
        vm["times"].as<int>(), 
        vm["en-size"].as<int>(),
        vm["ob-num"].as<int>(),
        vm["init-error"].as<double>()
        );

    return 0;
}