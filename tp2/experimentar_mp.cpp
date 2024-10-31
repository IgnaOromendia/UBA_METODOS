#include "metodo_potencia.cpp"

// Para compilar
// g++ -w -O3 -std=c++17 -I ./eigen-3.4.0/ experimentar_mp.cpp -o ejecMP

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Falta el archivo o cantidad de autovalores" << endl;
        return 1;
    }

    string input_file  = argv[1]; 
    string output_file = argv[2];
    bool experimentar  = string(argv[3]) == "1";

    if (experimentar) {
        vector<vector<MatrixXd>> matrices = leer_input_para_experimentar(input_file); // Por cada epsilon un vector de Matrices
        vector<vector<autoData>> results;
        procesar_matrices_para_experimentar(matrices, results);
        escribir_output_experimento(results, output_file);
    } else {
        vector<MatrixXd> matrices = leer_input(input_file);
        vector<vector<autoData>> results;
        procesar_matrices(matrices, results);
        escribir_output(results, output_file);
    }    
}