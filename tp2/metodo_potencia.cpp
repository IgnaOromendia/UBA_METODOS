#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef pair<double, VectorXd> AutoValVec;

struct AutoData {
    double l;
    VectorXd v;
    int it;
    double err;
    AutoData(double autoval, VectorXd autovec, int i, double error): l(autoval), v(autovec), it(i), err(error) {}
};

AutoData metodo_potencia(MatrixXd& A, int niter, double eps) {
    int n = A.rows();
    double l = 1;   
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

    l = (v.transpose() * A * v).value() / (v.transpose() * v).value();    

    double err = (A * v - l * v).lpNorm<2>();
    
    return AutoData(l, v, i, err);
}

vector<AutoData> obtener_autovalores(MatrixXd& A, int num, int niter=10e8, double eps=10e-7) {
    MatrixXd B = A;
    vector<AutoData> result;

    for (int i = 0; i < num; i++) {
        AutoData res = metodo_potencia(B, niter, eps);
        result.push_back(res);
        B = B - res.l * (res.v * res.v.transpose());
    }
    
    return result;
}

void escribir_output(vector<vector<AutoData>>& resultados) {
    ofstream f("output_mp.out");

    int cant_autovalores = resultados[0].size();

    f << resultados.size() << endl;
    f << cant_autovalores << endl;
    f << cant_autovalores + (resultados[0][0].v.size() * cant_autovalores) << endl;

    for(auto& res: resultados) {
        for (int i = 0; i < res.size(); i++) {
            f << res[i].l << " " << res[i].it << " " << res[i].err << endl;
            f << res[i].v << endl;
        }
    }
    
    f.close();
}

vector<MatrixXd> leer_input(string nombre_archivo) {
    ifstream f(nombre_archivo);

    if (!f) {
        cerr << "Error en el archivo";
        exit(1);
    }

    int cant_mat; f >> cant_mat;
    vector<MatrixXd> result;

    for (int i = 0; i < cant_mat; i++) {
        int filas, columnas;

        f >> filas >> columnas;

        MatrixXd M(filas, columnas);

        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                f >> M(i, j);  
            }
        }

        result.push_back(M);
    }

    return result;
}

// Para compilar
// g++ -w -O3 -std=c++17 -I ./eigen-3.4.0/ metodo_potencia.cpp -o ejecMP

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Falta el archivo o cantidad de autovalores" << endl;
        return 1;
    }

    vector<MatrixXd> matrices = leer_input(argv[1]);
    
    vector<vector<AutoData>> results;

    for(MatrixXd A: matrices)
        results.push_back(obtener_autovalores(A, stod(argv[2])));

    escribir_output(results);
}