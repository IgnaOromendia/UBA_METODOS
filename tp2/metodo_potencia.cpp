#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

typedef pair<vector<double>, vector<vector<double> > > EigensValue;

void power_iteartion(MatrixXd& A, int niter, double eps) {
    int n = A.rows();
    double lambda = 1;   
    VectorXd v(n); v.setConstant(-1);
    
    for (int i=0; i < niter; i++){
        VectorXd Av = A * v;
        v = Av / Av.norm();
        // Criterio
    };

    // a = (v.T @ A @ v) / (v.T @ v)
    lambda = (v.transpose() * A * v).value() / (v.transpose() * v).value();    

    //return  
}