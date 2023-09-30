#pragma once
#include "utility.hpp"
#include <assert.h>

namespace shiki
{

    class SweBHMM : public BHMM
    {
        public:
        enum class BoundaryCondition
        {
            Periodic,
            Reflective,
            Open
        };

        class SweSetting
        {
        public:
            double H;
            double g;
            double kappa;
            arma::vec f;
        };

        class Structure2d
        {
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
            Structure2d() = default;

            Structure2d(
                double left,
                double right,
                int grid_x,
                double low,
                double high,
                int grid_y,
                int unknowns,
                BoundaryCondition bc = BoundaryCondition::Reflective)
                // init class members
                : m_left(left), m_right(right), m_low(low), m_high(high), m_grid_x(grid_x), m_grid_y(grid_y), m_unknowns(unknowns), m_bc(bc)
            {
                m_dx = (m_right - m_left) / (m_grid_x - 1);
                m_dy = (m_high - m_low) / (m_grid_y - 1);

                int correction = 0;
                if (m_bc == BoundaryCondition::Periodic)
                {
                    correction = -1;
                }
                m_grid_x += correction;
                m_grid_y += correction;
            }

            void allocate_fields(arma::vec &fields)
            {
                fields.set_size(m_grid_x * m_grid_y * m_unknowns);
            }

            inline double &operator()(arma::vec &f, int i, int j)
            {
                // assert(i >= 0 && i < m_grid_y);
                // assert(j >= 0 && j < m_grid_x);
                return f(j * m_grid_y + i);
            }

            // overload () operator to access fields
            // i: x index
            // j: y index
            // k: field index
            // x axis is flipped
            inline double &operator()(arma::vec &fields, int i, int j, int k)
            {
                // assert(i >= 0 && i < m_grid_y);
                // assert(j >= 0 && j < m_grid_x);
                // assert(k >= 0 && k < m_unknowns);
                return fields(j * m_grid_y * m_unknowns + i * m_unknowns + k);
            }

            // overload () operator to set jacobian
            inline void operator()(
                arma::sp_mat &jacobian,
                int fi, int fj, int fk,
                int yi, int yj, int yk,
                double value)
            {
                // assert(fi >= 0 && fi < m_grid_y);
                // assert(fj >= 0 && fj < m_grid_x);
                // assert(fk >= 0 && fk < m_unknowns);
                // assert(yi >= 0 && yi < m_grid_y);
                // assert(yj >= 0 && yj < m_grid_x);
                // assert(yk >= 0 && yk < m_unknowns);
                jacobian(fj * m_grid_y * m_unknowns + fi * m_unknowns + fk, yj * m_grid_y * m_unknowns + yi * m_unknowns + yk) = value;
            }

            inline auto operator()(
                arma::sp_mat &jacobian,
                int fi, int fj, int fk,
                int yi, int yj, int yk)
            {
                // assert(fi >= 0 && fi < m_grid_y);
                // assert(fj >= 0 && fj < m_grid_x);
                // assert(fk >= 0 && fk < m_unknowns);
                // assert(yi >= 0 && yi < m_grid_y);
                // assert(yj >= 0 && yj < m_grid_x);
                // assert(yk >= 0 && yk < m_unknowns);
                return jacobian(fj * m_grid_y * m_unknowns + fi * m_unknowns + fk, yj * m_grid_y * m_unknowns + yi * m_unknowns + yk);
            }

            double average_h(arma::vec &fields)
            {
                double sum = 0;
                for (int j = 0; j < m_grid_x; j++)
                {
                    for (int i = 0; i < m_grid_y; i++)
                    {
                        sum += (*this)(fields, i, j, 2);
                    }
                }
                return sum / (m_grid_x * m_grid_y);
            }

            inline int index2int(int i, int j, int k)
            {
                return j * m_grid_y * m_unknowns + i * m_unknowns + k;
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
            arma::vec &Y,
            arma::vec &Ydot,
            Structure2d &s)
        {
            double dx = s.m_dx;
            double dy = s.m_dy;
            arma::vec &f = s.m_setting.f;
            double g = s.m_setting.g;
            double H = s.m_setting.H;
            double kappa = s.m_setting.kappa;

            using std::pow;
            arma::vec i_left(Y.n_rows, arma::fill::none);

            // write a lambda function to access normal fields or ghost fields
            auto access = [&s](arma::vec &f, int i, int j, int k)
            {
                double flip = 1;
                if (i < 0)
                {
                    i = 1;
                    if (k == 1)
                    {
                        flip = -1;
                    }
                }
                else if (i >= s.m_grid_y)
                {
                    i = s.m_grid_y - 2;
                    if (k == 1)
                    {
                        flip = -1;
                    }
                }
                else if (j < 0)
                {
                    j = 1;
                    if (k == 0)
                    {
                        flip = -1;
                    }
                }
                else if (j >= s.m_grid_x)
                {
                    j = s.m_grid_x - 2;
                    if (k == 0)
                    {
                        flip = -1;
                    }
                }

                return flip * s(f, i, j, k);
            };

            for (int j = 0; j < s.m_grid_x; j++)
            {
                for (int i = 0; i < s.m_grid_y; i++)
                {
                    double u = access(Y, i, j, 0);
                    double v = access(Y, i, j, 1);
                    double eta = access(Y, i, j, 2);
                    double udot = access(Ydot, i, j, 0);
                    double vdot = access(Ydot, i, j, 1);
                    double etadot = access(Ydot, i, j, 2);

                    double deta_dx = (access(Y, i, j + 1, 2) - access(Y, i, j - 1, 2)) / (2 * dx);
                    double deta_dy = (access(Y, i - 1, j, 2) - access(Y, i + 1, j, 2)) / (2 * dy);

                    double detaHu_dx = 1. / (2 * dx) * ((access(Y, i, j + 1, 2) + s.m_setting.H) * access(Y, i, j + 1, 0) - (access(Y, i, j - 1, 2) + s.m_setting.H) * access(Y, i, j - 1, 0));
                    double detaHv_dy = 1. / (2 * dy) * ((access(Y, i - 1, j, 2) + s.m_setting.H) * access(Y, i - 1, j, 1) - (access(Y, i + 1, j, 2) + s.m_setting.H) * access(Y, i + 1, j, 1));

                    s(i_left, i, j, 0) = udot - s(f, i, j) * v + g * deta_dx + kappa * u;
                    s(i_left, i, j, 1) = vdot + s(f, i, j) * u + g * deta_dy + kappa * v;
                    s(i_left, i, j, 2) = etadot + detaHu_dx + detaHv_dy;
                }
            }

            return i_left;
        }

        arma::sp_mat form_Jacobian_Y(
            double t,
            arma::vec &Y,
            arma::vec &Ydot,
            Structure2d &s)
        {
            double dx = s.m_dx;
            double dy = s.m_dy;
            arma::vec &f = s.m_setting.f;
            double g = s.m_setting.g;
            double H = s.m_setting.H;
            double kappa = s.m_setting.kappa;

            arma::sp_mat jacobian(Y.n_rows, Y.n_rows);

            auto access = [&s](arma::vec &f, int i, int j, int k)
            {
                double flip = 1;
                if (i < 0)
                {
                    i = 1;
                    if (k == 1)
                    {
                        flip = -1;
                    }
                }
                else if (i >= s.m_grid_y)
                {
                    i = s.m_grid_y - 2;
                    if (k == 1)
                    {
                        flip = -1;
                    }
                }
                else if (j < 0)
                {
                    j = 1;
                    if (k == 0)
                    {
                        flip = -1;
                    }
                }
                else if (j >= s.m_grid_x)
                {
                    j = s.m_grid_x - 2;
                    if (k == 0)
                    {
                        flip = -1;
                    }
                }

                return flip * s(f, i, j, k);
            };

            auto add = [&s](
                           arma::sp_mat &jacobian,
                           int fi, int fj, int fk,
                           int yi, int yj, int yk,
                           double val)
            {
                if (yi < 0)
                {
                    yi = 1;
                    if (yk == 1)
                        val *= -1;
                }
                else if (yi >= s.m_grid_y)
                {
                    yi = s.m_grid_y - 2;
                    if (yk == 1)
                        val *= -1;
                }
                else if (yj < 0)
                {
                    yj = 1;
                    if (yk == 0)
                        val *= -1;
                }
                else if (yj >= s.m_grid_x)
                {
                    yj = s.m_grid_x - 2;
                    if (yk == 0)
                        val *= -1;
                }
                s(jacobian, fi, fj, fk, yi, yj, yk) += val;
            };

            for (int j = 0; j < s.m_grid_x; j++)
            {
                for (int i = 0; i < s.m_grid_y; i++)
                {
                    add(jacobian, i, j, 0, i, j, 0, kappa);
                    add(jacobian, i, j, 0, i, j, 1, -s(f, i, j));
                    add(jacobian, i, j, 0, i, j + 1, 2, g / (2 * dx));
                    add(jacobian, i, j, 0, i, j - 1, 2, -g / (2 * dx));

                    add(jacobian, i, j, 1, i, j, 1, kappa);
                    add(jacobian, i, j, 1, i, j, 0, s(f, i, j));
                    add(jacobian, i, j, 1, i + 1, j, 2, -g / (2 * dy));
                    add(jacobian, i, j, 1, i - 1, j, 2, g / (2 * dy));

                    add(jacobian, i, j, 2, i, j + 1, 0, 1. / (2 * dx) * (access(Y, i, j + 1, 2) + H));
                    add(jacobian, i, j, 2, i, j + 1, 2, 1. / (2 * dx) * access(Y, i, j + 1, 0));
                    add(jacobian, i, j, 2, i, j - 1, 0, -1. / (2 * dx) * (access(Y, i, j - 1, 2) + H));
                    add(jacobian, i, j, 2, i, j - 1, 2, -1. / (2 * dx) * access(Y, i, j - 1, 0));
                    add(jacobian, i, j, 2, i - 1, j, 1, 1. / (2 * dy) * (access(Y, i - 1, j, 2) + H));
                    add(jacobian, i, j, 2, i - 1, j, 2, 1. / (2 * dy) * access(Y, i - 1, j, 1));
                    add(jacobian, i, j, 2, i + 1, j, 1, -1. / (2 * dy) * (access(Y, i + 1, j, 2) + H));
                    add(jacobian, i, j, 2, i + 1, j, 2, -1. / (2 * dy) * access(Y, i + 1, j, 1));
                }
            }

            return jacobian;
        }

        arma::sp_mat form_Jacobian_Ydot(
            double t,
            arma::vec &Y,
            arma::vec &Ydot,
            Structure2d &s)
        {
            double dx = s.m_dx;
            double dy = s.m_dy;

            arma::sp_mat jacobian(Y.n_rows, Y.n_rows);

            for (int j = 0; j < s.m_grid_x; j++)
            {
                for (int i = 0; i < s.m_grid_y; i++)
                {
                    s(jacobian, i, j, 0, i, j, 0) += 1;
                    s(jacobian, i, j, 1, i, j, 1) += 1;
                    s(jacobian, i, j, 2, i, j, 2) += 1;
                }
            }

            return jacobian;
        }

        // function solver for shallow water equation, using rk2
        arma::vec model(
            double t,
            double dt,
            const arma::vec &Y,
            Structure2d &structure,
            double tol = 1e-6)
        {
            // tol *= structure.m_grid_x * structure.m_grid_y;
            // lambda function to calculate loss
            // using t, Y, Ydot, structure
            auto form_loss = [this, dt](double t, arma::vec &Y, arma::vec &Ydot, Structure2d &structure) -> arma::vec
            {
                arma::vec i_left = form_Ifunction(t, Y, Ydot, structure);
                return i_left;
            };

            arma::vec Y_next = Y;
            int try_count = 0;

            while (true)
            {
                arma::vec Ydot = (Y_next - Y) / dt;
                arma::vec Y_mid = (Y_next + Y) / 2;
                arma::vec loss_vec = form_loss(t + dt / 2, Y_mid, Ydot, structure);
                double loss = arma::norm(loss_vec, 2);

                // std::cout<<Y_next<<loss_vec;
                // std::cout<<"time: "<<t<<" time stepping "<<try_count++<<", loss is "<<loss<<std::endl;

                if (loss < tol)
                {
                    break;
                }

                arma::sp_mat jacobian1 = form_Jacobian_Y(t + dt / 2, Y_mid, Ydot, structure);
                arma::sp_mat jacobian2 = form_Jacobian_Ydot(t + dt / 2, Y_mid, Ydot, structure);
                arma::sp_mat jacobian = 0.5 * jacobian1 + 1. / dt * jacobian2;
                // std::cout<<jacobian<<std::endl;
                // std::cout<<(arma::mat)jacobian<<std::endl;
                Y_next += shiki::Krylov(jacobian, -loss_vec);
            }

            return Y_next;
        }

        double init_u(double x, double y)
        {
            return 0;
        }

        double init_v(double x, double y)
        {
            return 0;
        }

        void init(arma::vec &uvh, Structure2d &s, bool random = false, double init_error = 0.03)
        {
            double cx = 1e6 / 2.7, cy = 1e6 / 4.;
            double sx = 5e4, sy = 5e4;
            if (random)
            {
                cx *= 1 + arma::randn() * init_error;
                cy *= 1 + arma::randn() * init_error;
                sx *= 1 + arma::randn() * init_error;
                sy *= 1 + arma::randn() * init_error;
            }
            auto init_h = [&cx, &cy, &sx, &sy](double x, double y)
            {
                return std::exp(
                    -(
                        pow(x - cx, 2) / (2 * pow(sx, 2)) + pow(y - cy, 2) / (2 * pow(sy, 2))));
            };

            for (int j = 0; j < s.m_grid_x; ++j)
            {
                double x = s.m_left + j * s.m_dx;
                for (int i = 0; i < s.m_grid_y; ++i)
                {
                    double y = s.m_high - i * s.m_dy;
                    s(uvh, i, j, 0) = init_u(x, y);
                    s(uvh, i, j, 1) = init_v(x, y);
                    s(uvh, i, j, 2) = init_h(x, y);
                }
            }
        }

        void init_setting(Structure2d &s)
        {
            s.m_setting.g = 9.8;
            s.m_setting.H = 100;
            s.m_setting.kappa = 1. / (5. * 24 * 3600);

            s.m_setting.f.set_size(s.m_grid_x * s.m_grid_y);
            for (int j = 0; j < s.m_grid_x; ++j)
            {
                double x = s.m_left + j * s.m_dx;
                for (int i = 0; i < s.m_grid_y; ++i)
                {
                    double y = s.m_high - i * s.m_dy;
                    s(s.m_setting.f, i, j) = 1e-4 + 2e-11 * y;
                }
            }
        }

    public:
        double t0, t, dt;
        double L;
        int iter_times;
        std::vector<double> times;
        Structure2d structure;
        vec sol;

    public:
        SweBHMM(int gx, int gy, int iter_times = 5000)
        {
            t0 = 0; t = t0;
            L = 1e6;
            this->iter_times = iter_times;
            structure = Structure2d(-L / 2, L / 2, gx, -L / 2, L / 2, gy, 3);
            init_setting(structure);
            dt = 0.1 * L / std::max(gx - 1, gy - 1) /
                 std::sqrt(structure.m_setting.g * structure.m_setting.H);

            structure.allocate_fields(sol);
            init(sol, structure);
            times.push_back(t);
            sol.save("./data/swe/sol" + std::to_string(t) + ".bin", arma::raw_binary);
        }

        std::vector<double> get_times() override
        {
            return times;
        }

        void assimilate() override
        {
            for (int i = 0; i < iter_times; ++i)
            {
                sol = model(t, dt, sol, structure);
                t += dt;
                times.push_back(t);
                sol.save("./data/swe/sol" + std::to_string(t) + ".bin", arma::raw_binary);
            }
        }

        vec get_state(double t) override
        {
            // update t to closest value in times
            int idx = std::lower_bound(times.begin(), times.end(), t) - times.begin();
            // read from file
            vec state;
            state.load("./data/swe/sol" + std::to_string(times[idx]) + ".bin", arma::raw_binary);
            return state;
        }

        mat model(double t, double dt, const mat &e) override
        {
            mat next = mat(e.n_rows, e.n_cols, arma::fill::none);
            for (int i = 0; i < e.n_cols; ++i)
            {
                next.col(i) = model(t, dt, e.col(i), structure);
            }
            return next;
        }

        sp_mat noise(double t) override
        {
            return sp_mat(sol.n_rows, sol.n_rows);
        }
    };

    class LinearObserver : public Observer
    {
    public:
        int gap;
        std::vector<double> times;
        std::vector<sp_mat> Hs;
        std::vector<mat> noises;
        std::vector<vec> observations;
        BHMM &hmm;

    public:
        LinearObserver(int gap, BHMM &hmm) : gap(gap), hmm(hmm)
        {
            auto possible_times = hmm.get_times();
            for (int i = 1; i < possible_times.size(); i += gap)
            {
                times.push_back(possible_times[i]);
            }
        }

        auto get_times()
        {
            return times;
        }

        void set_H_noise(std::vector<sp_mat> &Hs, std::vector<mat> &noises)
        {
            this->Hs = Hs;
            this->noises = noises;
            assert(Hs.size() == times.size());
            assert(noises.size() == times.size());
        }

        bool is_observable(double t) override
        {
            return std::binary_search(times.begin(), times.end(), t);
        }

        void observe() override
        {
            for (int i = 0; i < times.size(); ++i)
            {
                vec real = hmm.get_state(times[i]);
                vec noise = arma::mvnrnd(vec(noises[i].n_rows, arma::fill::zeros), noises[i]);
                // if(arma::norm(noise) != 0){
                //     std::cout<<"noise is not zero"<<std::endl;
                // }
                observations.push_back(Hs[i] * real + noise);
            }
        }

        vec get_observation(double t) override
        {
            int idx = std::lower_bound(times.begin(), times.end(), t) - times.begin();
            return observations.at(idx);
        }

        vec observe(double t, vec state) override
        {
            int idx = std::lower_bound(times.begin(), times.end(), t) - times.begin();
            return Hs[idx] * state;
        }

        mat observe(double t, mat state) override
        {
            int idx = std::lower_bound(times.begin(), times.end(), t) - times.begin();
            return Hs[idx] * state;
        }

        sp_mat linear(double t, vec state) override
        {
            int idx = std::lower_bound(times.begin(), times.end(), t) - times.begin();
            return Hs[idx];
        }

        mat noise(double t) override
        {
            int idx = std::lower_bound(times.begin(), times.end(), t) - times.begin();
            return noises[idx];
        }
    };
}