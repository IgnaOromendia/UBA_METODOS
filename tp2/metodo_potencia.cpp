#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef pair<double, VectorXd> AutoValVec;

struct AutoData {
    double l;
    VectorXd v;
    int it;
    AutoData(double autoval, VectorXd autovec, int i): l(autoval), v(autovec), it(i) {}
};


AutoData metodo_potencia(MatrixXd& A, int niter, double eps) {
    int n = A.rows();
    double lambda = 1;   
    VectorXd v(n); v.setConstant(-1);
    VectorXd w = v;
    int i=0;
    
    for (; i < niter; i++){
        VectorXd Av = A * v;
        v = Av / Av.norm();

        // Criterio de parada
        if ((v - w).lpNorm<Infinity>() < eps) break;

        w = v;
    };

    lambda = (v.transpose() * A * v).value() / (v.transpose() * v).value();    

    return AutoData(lambda, v, i);
}

vector<AutoData> obtener_autovalores(MatrixXd& A, int num, int niter=10000, double eps=1e-6) {
    MatrixXd B = A;
    vector<AutoData> result;

    for (int i = 0; i < num; i++) {
        AutoData res = metodo_potencia(B, niter, eps);
        result.push_back(res);
        B = B - res.l * (res.v * res.v.transpose());
    }
    
    return result;
}

