import numpy as np

def generar_matriz_spd(n, nombre_archivo):
    print(f"Generando matriz de {n}x{n}...")
    
    # creamos una matriz aleatoria B
    B = np.random.rand(n, n)
    
    # multiplicar B por su transpuesta garantiza que sea simétrica
    A = np.dot(B, B.T)
    
    # sumar a la diagonal garantiza que sea estrictamente definida positiva
    # garantiza que la matriz sea invertible, es decir que su determinante no sea 0
    # gracias a que hace que sus valores propios tiendan a ser positivos
    A += np.eye(n) * 1e-3 


    
    print(f"Guardando en {nombre_archivo}...")
    with open(nombre_archivo, 'w') as f:
        f.write(f"{n}\n")
        for i in range(n):
            # guardamos con 6 decimales separados por espacio
            fila = " ".join(f"{val:.6f}" for val in A[i])
            f.write(fila + "\n")
            
    print(f"{nombre_archivo} completado\n")



dimension = 100
nombre_matriz = f'matriz_{dimension}.txt'
generar_matriz_spd(dimension, nombre_matriz)
