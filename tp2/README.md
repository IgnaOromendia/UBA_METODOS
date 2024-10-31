# TP2

### Requisitos
Tener la carpeta **eigen-3.4.0** en el directorio del trabajo práctico

### Instrucciones
Compilar el archivo que contiene el método de la potencia
```bash
g++ -w -O3 -std=c++17 -I ./eigen-3.4.0 experimentar_mp.cpp -o ejecMP
```
Luego ejecutar el siguiente comando para plotear los experimentos como la convergencia y los erroes del método de la potencia, entre otros.
```bash
python experimentos.py
```

Para compilar el tester del método de la potencia
```bash
g++ -w -O3 -std=c++17 -I ./eigen-3.4.0 tests.cpp -o ejecTest
```

Para ejecutar el tester
```bash
./ejecTest
```