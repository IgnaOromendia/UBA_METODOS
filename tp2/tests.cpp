#include "metodo_potencia.cpp"
#include <cassert>
#include <string>

// Para compilar y ejecutar
// g++ -w -O3 -std=c++17 -I ./eigen-3.4.0/ tests.cpp -o ejecTest;./ejecTest

MatrixXd matriz_householder(VectorXd w) {
    int n = w.size();
    MatrixXd D = MatrixXd::Zero(n,n);

    D.diagonal() = w;

    VectorXd v(n); v.setConstant(1); // Randomizar esto

    v = v / v.lpNorm<2>();

    MatrixXd H = MatrixXd::Identity(n,n) - 2 * (v * v.transpose());

    return H * D * H.transpose();
}

void test01_metodo_potencia() {
    VectorXd v(5); v << 5.0, 4.0, 3.0, 2.0, 1.0;
    MatrixXd A = matriz_householder(v);

    autoData result = metodo_potencia(A, 10000, 1e-20);

    assert(abs(result.l - 5.0) < 1e-6);

    cout << "Test 1 Método Potencia OK" << endl;
}

void test02_deflacion_diagonal() {
    VectorXd v(5); v << 5.0, 4.0, 3.0, 2.0, 1.0;
    MatrixXd D = MatrixXd::Zero(5, 5);
    D.diagonal() = v;
    
    vector<autoData> res = obtener_autovalores(D, 10e2, 10e-7);

    for(int i = 0; i < 5; i++) {
        double l_esperado = v(i);
        double l_obtenido = res[i].l;
        assert(abs(l_obtenido - l_esperado) < 1e-6);
    }
    
    cout << "Test 2 Deflación con Diagonal OK" << endl;
}

void test03_deflacion_householder(){
    VectorXd v(10); v << 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0;
    MatrixXd B = matriz_householder(v);

    vector<autoData> res = obtener_autovalores(B, 10e2, 10e-7);

    for(int i = 0; i < 10; i++) {
        double l_esperado = v(i);
        double l_obtenido = res[i].l;
        assert(abs(l_obtenido - l_esperado) < 1e-6);
    }
    
    cout << "Test 3 Deflación con Householder OK" << endl;
}

int main() {
    test01_metodo_potencia();
    test02_deflacion_diagonal();
    test03_deflacion_householder();
}