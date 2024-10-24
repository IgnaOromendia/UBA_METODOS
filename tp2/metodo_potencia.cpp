#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef pair<double, VectorXd> AutoValVec;

struct NOMBRE {
    vector<double> autovalores;
    vector<VectorXd> autovectores;
    vector<int> iteraciones;
}

AutoValVec metodo_potencia(MatrixXd& A, int niter, double eps) {
    int n = A.rows();
    double lambda = 1;   
    VectorXd v(n); v.setConstant(-1);
    VectorXd w = v;
    
    for (int i=0; i < niter; i++){
        VectorXd Av = A * v;
        v = Av / Av.norm();

        // Criterio de parada
        if ((v - w).lpNorm<Infinity>() < eps) break;

        w = v;
    };

    lambda = (v.transpose() * A * v).value() / (v.transpose() * v).value();    

    return make_pair(lambda, v);
}

AutoValsVecs obtener_autovalores(MatrixXd& A, int num, int niter=10000, double eps=1e-6) {
    MatrixXd B = A;
    vector<double> autovalores;
    vector<VectorXd> autovectores;

    for (int i = 0; i < num; i++) {
        AutoValVec res = metodo_potencia(B, niter, eps);

        double l    = res.first;
        VectorXd v  = res.second; 

        autovalores.push_back(l);
        autovectores.push_back(v);

        
        B = B - l * (v * v.transpose());
    }
    
    return AutoValsVecs(autovalores, autovectores);
}

