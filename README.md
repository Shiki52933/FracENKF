# FracENKF
带分数阶导数的（随机）动力系统的（集合）卡尔曼滤波数据同化实现，部分Python慢的场合用C++实现

# 环境要求
sudo apt-get install libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev
sudo apt-get install libboost-dev
sudo apt-get install libarmadillo-dev

# 编译命令
g++ lorenz63.cpp -l armadillo -l boost_program_options -O2 -o lorenz63

g++ lorenz96.cpp -l armadillo -l boost_program_options -O2 -o lorenz96