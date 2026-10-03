import numpy as np
import sys

def generar_matriz_spd_binaria(n, nombre_archivo):
    print(f"Generando matriz SPD de {n}x{n}...")
    
    B = np.random.rand(n, n)
    A = np.dot(B, B.T)
    A += np.eye(n) * 1e-3 
    
    print(f"Guardando formato crudo en {nombre_archivo}...")
    # Se guarda el arreglo en disco en formato binario nativo (float64 = double en C)
    # No se guarda la dimensión N dentro del archivo para simplificar la lectura MPI
    A.tofile(nombre_archivo)
            
    print(f"{nombre_archivo} completado. Tamaño exacto: {n * n * 8} bytes.\n")

# Uso: python generar_matrices.py 100
if __name__ == "__main__":
    dimension = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    nombre_matriz = f'matriz_{dimension}.bin'
    generar_matriz_spd_binaria(dimension, nombre_matriz)