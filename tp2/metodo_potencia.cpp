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
    double it_promedio = 0;
    double it_desvio = 0;
    double err;
    double err_promedio = 0;
    double err_desvio = 0;
    AutoData(double autoval, VectorXd autovec, int i, double error): l(autoval), v(autovec), it(i), err(error) {}
};

double calcular_promedio(vector<double>& v) {
    double suma = 0;

    for(double e: v)
        suma += e;

    return suma / v.size();
}

double calcular_desvio(vector<double>& v, double promedio) {
    double sumaCuadrados = 0.0;

    for (double e : v) 
        sumaCuadrados += pow(e - promedio, 2);
    
    return sqrt(sumaCuadrados / v.size());
}

AutoData metodo_potencia(MatrixXd& A, int niter, double eps, int reps) {
    double l = 1;   
    VectorXd v(A.rows()); v.setConstant(-1);
    VectorXd w = v;
    int i = 0;

    for (; i < niter; i++){
        VectorXd Av = A * v;
        v = Av / Av.norm();

        // Criterio de parada
        if ((v - w).lpNorm<Infinity>() < eps) break;
        
        w = v;
    }

    l = (v.transpose() * A * v).value() / (v.transpose() * v).value();    

    double err = (A * v - l * v).lpNorm<2>();
    
    return AutoData(l, v, i, err);
}

vector<AutoData> obtener_autovalores(MatrixXd& A, int niter=10e8, double eps=10e-7, int reps=5) {
    MatrixXd B = A;
    vector<AutoData> result;

    for (int i = 0; i < B.rows(); i++) {
        AutoData res = metodo_potencia(B, niter, eps, reps);
        result.push_back(res);
        B = B - res.l * (res.v * res.v.transpose());
    }
    
    return result;
}

void escribir_vector(ofstream& f, VectorXd& vec) {
    for (int i = 0; i < vec.size(); ++i) 
        f << vec(i) << ", ";
}

void escribir_output(vector<vector<AutoData>>& res) {
    ofstream f("output_mp.csv");

    f << "Matriz,Autovalor,Autovector...,Iteraciones...,Error\n";

    for(int k = 0; k < res.size(); k++) {
        for (int i = 0; i < res[k].size(); i++) {
            f << k + 1 << ", " <<  res[k][i].l << ", ";
            escribir_vector(f, res[k][i].v);
            f << res[k][i].it_promedio << ", " << res[k][i].it_desvio << ", " << res[k][i].err_promedio << ", " << res[k][i].err_desvio << endl;
        }
    }
    
    f.close();
}

vector<vector<MatrixXd>> leer_input(string nombre_archivo) {
    ifstream f(nombre_archivo);

    if (!f) {
        cerr << "Error en el archivo";
        exit(1);
    }

    int cant_mat; f >> cant_mat;
    int cant_mat_por_eps; f >> cant_mat_por_eps;

    vector<vector<MatrixXd>> result(cant_mat);

    for (int i = 0; i < cant_mat; i++) {
        vector<MatrixXd> mat_por_eps(cant_mat_por_eps);

        for (int k = 0; k < cant_mat_por_eps; k++) { 
            int filas, columnas;

            f >> filas >> columnas;

            MatrixXd M(filas, columnas);

            for (int i = 0; i < filas; i++) {
                for (int j = 0; j < columnas; j++) {
                    f >> M(i, j);  
                }
            }

            mat_por_eps[k] = M;
        }
        

        result[i] = mat_por_eps;
    }

    return result;
}

// Para compilar
// g++ -w -O3 -std=c++17 -I ./eigen-3.4.0/ metodo_potencia.cpp -o ejecMP

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Falta el archivo o cantidad de autovalores" << endl;
        return 1;
    }

    vector<vector<MatrixXd>> matrices = leer_input(argv[1]);
    
    vector<vector<AutoData>> results;

    // Por cada matriz a experimentar vamos a promediar y calcular el desvio de todas las iteraciones y errores
    for(int i = 0; i < matrices.size(); i++) {
        vector<vector<double>> iteraciones(matrices[0][0].size()); // Iteraciones por cada lambda
        vector<vector<double>> errores(matrices[0][0].size()); // Errores por cada lambda
        
        vector<AutoData> autodata_por_eps; // Representante de cada eps (ya que todos tinenen los mismo autovalores)

        for(int k = 0; k < matrices[i].size(); k++) {
            vector<AutoData> data = obtener_autovalores(matrices[i][k]);
            
            // Guardamos las iteraciones de cada lambda
            for(int l = 0; l < data.size(); l++) {
                iteraciones[l].push_back(data[l].it);
                errores[l].push_back(data[l].err);
            }
                
            // Guardamos representate
            if (k == 0) autodata_por_eps = data; 
        }

        for(int l = 0; l < autodata_por_eps.size(); l++) {
            autodata_por_eps[l].it_promedio = calcular_promedio(iteraciones[l]);
            autodata_por_eps[l].it_desvio   = calcular_desvio(iteraciones[l], autodata_por_eps[l].it_promedio);

            autodata_por_eps[l].err_promedio = calcular_promedio(errores[l]);
            autodata_por_eps[l].err_desvio   = calcular_desvio(errores[l], autodata_por_eps[l].err_promedio);
        }

        results.push_back(autodata_por_eps);
        
    }
        

    escribir_output(results);
}