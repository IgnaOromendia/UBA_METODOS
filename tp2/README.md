# TP2

### Requisitos
Tener la carpeta **eigen-3.4.0** en el directorio del trabajo práctico

### Instrucciones
Compilar el archivo que contiene el método de la potencia
```bash
g++ -w -O3 -std=c++17 -I ./eigen-3.4.0 metodo_potencia.cpp -o ejecMP
```
Luego ejecutar el siguiente comando para plotear la convergencia y los erroes del método de la potencia
```bash
python experimentos.py
```