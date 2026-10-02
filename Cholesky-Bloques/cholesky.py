# =============================================================================
# Factorización de Cholesky (A = L·Lᵀ) — Proyecto 7, Métodos Numéricos II (HPC)
#
#   1) cholesky_literal      : fórmulas de la Unidad 3 tal cual, Python puro, sin ninguna optimización.
#   2) cholesky_optimizado   : secuencial por BLOQUES con NumPy (1 solo hilo). Aprovecha la caché.
#   3) cholesky_paralelizado : el mismo algoritmo por bloques, repartiendo las filas entre procesos MPI.
#
# Uso:
#   python cholesky.py --test                          -> valida las 3 versiones (paralela con 1 proceso)
#   mpiexec -n 4 python cholesky.py --test             -> valida la paralela con 4 procesos
#   python cholesky.py --serial                        -> benchmark literal + optimizado -> resultados_serial.csv
#   mpiexec -n 4 python cholesky.py --paralelo         -> benchmark paralelo con 4 procesos -> resultados_paralelo.csv
#   python cholesky.py --speedup                       -> tabla de speedup y eficiencia a partir de los 2 CSV
#   Opciones: --N 1000,2000,4000   --b 128   --reps 5
#
# Requisitos: numpy (obligatorio), scipy (recomendado), mpi4py + MPI (solo para la paralela).
#   En Windows: instalar Microsoft MPI (msmpisetup.exe) y luego "pip install mpi4py".
# =============================================================================

import os
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",   # Forzamos 1 hilo en BLAS ANTES de importar numpy:
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):               # si no, NumPy usa todos los núcleos "por detrás" y la
    os.environ[var] = "1"                                                   # versión secuencial no sería secuencial. Todo el paralelismo lo pone MPI.

import sys
import csv
import math
import time
import argparse
import statistics
import platform
import numpy as np

try:
    from scipy.linalg import solve_triangular                              # Resolución triangular eficiente (TRSM)
except ImportError:
    solve_triangular = None                                                 # Si no está scipy, usamos np.linalg.solve (más lento, mismo resultado)

try:
    from mpi4py import MPI                                                  # MPI para Python: mismas primitivas que en C (Bcast, Gather, Barrier...)
    COMM = MPI.COMM_WORLD
except ImportError:
    MPI = None
    COMM = None


# =============================================================================
# GENERACIÓN DE DATOS
# =============================================================================
def generar_filas_spd(n, filas):
    """Devuelve SOLO las filas pedidas de la matriz A (n×n) definida así:
         A[i][j] = 1 / (1 + |i - j|)      fuera de la diagonal
         A[i][i] = n                      en la diagonal
    Como cada elemento depende solo de i y j, cada proceso MPI puede generar
    únicamente sus filas, sin armar la matriz completa."""
    i = np.asarray(filas).reshape(-1, 1)                                    # Índices de las filas pedidas (como columna)
    j = np.arange(n)                                                        # Índices de todas las columnas 0..n-1
    A = 1.0 / (1.0 + np.abs(i - j))                                         # |i-j| = |j-i|  -> la matriz es SIMÉTRICA
    A[i == j] = n                                                           # Diagonal = n
    return A                                                                # Fuera de la diagonal cada valor es ≤ 1/2 y la suma
                                                                            # de una fila es < n -> diagonal dominante -> definida positiva


def generar_spd(n):
    """Matriz completa = todas sus filas."""
    return generar_filas_spd(n, np.arange(n))


# =============================================================================
# 1) CHOLESKY LITERAL — fórmulas de la Unidad 3, sin optimizar
#    l_ii = √( a_ii − Σ_{k<i} l_ik² )
#    l_ji = ( a_ji − Σ_{k<i} l_jk·l_ik ) / l_ii        para j = i+1 … n
# =============================================================================
def cholesky_literal(A):
    n = len(A)                                                              # Dimensión de la matriz (A es lista de listas)
    L = [[0.0 for _ in range(n)] for _ in range(n)]                         # L llena de ceros
    for i in range(n):                                                      # Recorremos cada columna i de izquierda a derecha
        suma = 0.0
        for k in range(i):                                                  # Σ l_ik² : cuadrados a la izquierda de la diagonal
            suma = suma + L[i][k] ** 2
        L[i][i] = math.sqrt(A[i][i] - suma)                                 # Elemento diagonal (si A no es DP, sqrt da error)
        for j in range(i + 1, n):                                           # Cada fila j debajo de la diagonal
            suma = 0.0
            for k in range(i):                                              # Σ l_jk·l_ik : fila j por fila i
                suma = suma + L[j][k] * L[i][k]
            L[j][i] = (A[j][i] - suma) / L[i][i]                            # Divide cada vez (sin guardar 1/l_ii): tosco a propósito
    return L


# =============================================================================
# 2) CHOLESKY OPTIMIZADO — secuencial, por bloques, NumPy con 1 hilo
#
#  Idea: en vez de avanzar de a 1 columna, avanzamos de a b columnas (un "bloque").
#  Así la mayoría de las cuentas son multiplicaciones de matrices chicas (b×b) que
#  entran en la caché del procesador y NumPy/BLAS hace a máxima velocidad.
#  Por cada bloque k (filas/columnas r0..r1):
#     a) Actualizar el bloque diagonal:  C = A_kk − L_k,prev · L_k,prevᵀ
#     b) Factorizarlo:                   L_kk = cholesky(C)       (bloque chico)
#     c) Actualizar el panel de abajo:   C_i = A_ik − L_i,prev · L_k,prevᵀ
#     d) Resolver triangular:            L_ik = C_i · L_kk⁻ᵀ      (TRSM)
#  Todo se hace IN-PLACE: L se guarda sobre el triángulo inferior de A.
# =============================================================================
def cholesky_bloque_diag(C):
    """Cholesky de un bloque chico b×b, in-place, columna a columna (Crout vectorizado).
    Es la misma fórmula que el literal, pero cada suma es un producto escalar de NumPy."""
    m = C.shape[0]
    for j in range(m):                                                      # Columna j del bloque
        Lj = C[j, :j]                                                       # Parte ya calculada de la fila j
        d = C[j, j] - Lj @ Lj                                               # a_jj − Σ l_jk²   (producto escalar en C, no en Python)
        if d <= 0:
            raise ValueError("A no es definida positiva")
        C[j, j] = math.sqrt(d)                                              # l_jj
        if j + 1 < m:                                                       # Todas las filas de abajo de UNA vez (vector):
            C[j + 1:, j] = (C[j + 1:, j] - C[j + 1:, :j] @ Lj) / C[j, j]    # l_ij = (a_ij − fila_i·fila_j) / l_jj
    return np.tril(C)                                                       # Nos quedamos con el triángulo inferior


def trsm(Lkk, C):
    """Devuelve X tal que X · Lkkᵀ = C  (o sea X = C · Lkk⁻ᵀ), sin calcular la inversa."""
    if solve_triangular is not None:
        return solve_triangular(Lkk, C.T, lower=True, check_finite=False).T # Sustitución hacia adelante (aprovecha que es triangular)
    return np.linalg.solve(Lkk, C.T).T                                      # Alternativa sin scipy


def cholesky_optimizado(A, b=128):
    n = A.shape[0]                                                          # A es un array NumPy n×n (se sobrescribe)
    for r0 in range(0, n, b):                                               # Recorremos bloques de b columnas de izq. a derecha
        r1 = min(r0 + b, n)                                                 # Último bloque puede ser más chico
        Lk_prev = A[r0:r1, :r0]                                             # Fila de bloques k, columnas ya factorizadas (vista, no copia)
        A[r0:r1, r0:r1] -= Lk_prev @ Lk_prev.T                              # a) bloque diagonal − contribución de columnas anteriores
        Lkk = cholesky_bloque_diag(A[r0:r1, r0:r1])                         # b) Cholesky del bloque diagonal (chico, entra en caché)
        A[r0:r1, r0:r1] = Lkk                                               #    lo guardamos en su lugar
        if r1 < n:                                                          # Si hay filas debajo del bloque:
            C = A[r1:, r0:r1] - A[r1:, :r0] @ Lk_prev.T                     # c) panel − contribución anterior (multiplicación de matrices)
            A[r1:, r0:r1] = trsm(Lkk, C)                                    # d) L_ik = C · L_kk⁻ᵀ
    return A                                                                # Triángulo inferior = L (el superior queda con basura)


# =============================================================================
# 3) CHOLESKY PARALELIZADO — mismo algoritmo por bloques, con MPI
#
#  Reparto: DISTRIBUCIÓN CÍCLICA por bloques de filas. El bloque de filas g es del
#  proceso g % p. Cada proceso guarda SOLO sus filas (memoria / p).
#  ¿Por qué cíclica? Porque a medida que avanzamos quedan menos filas por calcular;
#  si repartiéramos en tramos contiguos, los primeros procesos se quedarían sin trabajo.
#
#  Por cada bloque k:
#     1. El DUEÑO del bloque k hace a) y b): factoriza el bloque diagonal  (parte secuencial)
#     2. MPI_Bcast de su fila de bloques de L a todos                      (comunicación)
#     3. TODOS hacen c) y d) sobre sus propias filas de abajo, a la vez    (parte paralela, ~todo el cómputo)
# =============================================================================
def filas_de_proceso(n, b, rank, p):
    """Bloques de filas que le tocan a este proceso y sus índices globales de fila."""
    bloques = list(range(rank, math.ceil(n / b), p))                        # Bloques rank, rank+p, rank+2p, ...
    filas = [f for g in bloques for f in range(g * b, min(g * b + b, n))]   # Filas globales de esos bloques
    return bloques, np.array(filas, dtype=np.int64)


def cholesky_paralelizado(mis_filas, n, tam_bloque, comm):
    # ---------------------------------------------------------------- PREPARACIÓN
    yo = comm.Get_rank()                                    # Mi número de proceso: 0, 1, 2...
    cant_procesos = comm.Get_size()                         # Cuántos procesos somos en total
    total_bloques = math.ceil(n / tam_bloque)               # En cuántos bloques se corta la matriz (redondeando para arriba)

    mis_bloques, _ = filas_de_proceso(n, tam_bloque, yo, cant_procesos)   # Qué bloques de filas me tocaron (repartidos como cartas)

    # mis_filas tiene MIS filas una pegada a la otra, sin huecos.
    # Este diccionario dice: "el bloque X empieza en tal renglón de mis_filas".
    donde_empieza = {}
    renglon = 0
    for bloque in mis_bloques:
        donde_empieza[bloque] = renglon                     # Anoto dónde arranca este bloque
        alto = min((bloque + 1) * tam_bloque, n) - bloque * tam_bloque   # Cuántas filas tiene (el último puede ser más bajo)
        renglon += alto                                     # El siguiente bloque arranca justo debajo

    # ---------------------------------------------------------------- UN TURNO POR CADA BLOQUE
    for k in range(total_bloques):                          # Turnos 0, 1, 2... de izquierda a derecha (no se puede saltear: cada turno usa los anteriores)
        col_inicio = k * tam_bloque                         # Primera columna de este bloque
        col_fin = min(col_inicio + tam_bloque, n)           # Columna donde termina (sin incluirla)
        alto_k = col_fin - col_inicio                       # Cuántas filas/columnas tiene el bloque k
        dueno = k % cant_procesos                           # A quién le tocó el bloque k (el mismo reparto de cartas)

        fila_k = np.empty((alto_k, col_fin))                # "Sobre" vacío donde voy a recibir la fila de bloques k ya resuelta

        # ---- PASO 1: SOLO EL DUEÑO resuelve el bloque de la diagonal (los demás esperan)
        if yo == dueno:
            filas_k = mis_filas[donde_empieza[k] : donde_empieza[k] + alto_k]   # Mis filas del bloque k
            ya_calculado = filas_k[:, :col_inicio]          # Lo que está a la izquierda de la diagonal (ya es L)
            diagonal = filas_k[:, col_inicio:col_fin]       # El cuadrado de la diagonal (todavía es A)
            diagonal -= ya_calculado @ ya_calculado.T       # a) Le resto lo que aportan las columnas anteriores
            filas_k[:, col_inicio:col_fin] = cholesky_bloque_diag(diagonal)   # b) Cholesky chiquito de ese cuadrado
            fila_k[:] = filas_k[:, :col_fin]                # Meto en el sobre la fila de bloques k ya terminada

        # ---- PASO 2: EL DUEÑO MANDA EL SOBRE A TODOS (MPI_Bcast)
        comm.Bcast(fila_k, root=dueno)                      # Después de esta línea, TODOS tienen el mismo sobre
        L_izquierda = fila_k[:, :col_inicio]                # Parte del sobre a la izquierda de la diagonal
        L_diagonal = fila_k[:, col_inicio:col_fin]          # El cuadrado de la diagonal ya resuelto

        # ---- PASO 3: CADA UNO resuelve SUS filas de abajo, TODOS AL MISMO TIEMPO
        primer_bloque_abajo = None
        for bloque in mis_bloques:                          # Busco mi primer bloque que esté DEBAJO del bloque k
            if bloque > k:
                primer_bloque_abajo = bloque
                break                                       # Encontré uno: dejo de buscar
        if primer_bloque_abajo is None:                     # No tengo nada debajo: en este turno no trabajo
            continue

        filas_abajo = mis_filas[donde_empieza[primer_bloque_abajo]:]   # Todas mis filas desde ahí hasta el final
        resto = filas_abajo[:, col_inicio:col_fin] - filas_abajo[:, :col_inicio] @ L_izquierda.T   # c) Le resto lo que aportan las columnas anteriores
        filas_abajo[:, col_inicio:col_fin] = trsm(L_diagonal, resto)   # d) "Divido" por el cuadrado de la diagonal

    return mis_filas                                        # Mis filas de L terminadas (cada proceso devuelve solo las suyas)


def juntar_L(Aloc, n, b, comm):
    """Junta en el proceso 0 las filas de todos para armar L completa (solo para validar, N chico)."""
    _, filas = filas_de_proceso(n, b, comm.Get_rank(), comm.Get_size())
    partes = comm.gather((filas, Aloc), root=0)                             # Cada proceso manda (sus índices, sus filas)
    if comm.Get_rank() != 0:
        return None
    L = np.empty((n, n))
    for f, bloque in partes:
        L[f] = bloque                                                       # Cada fila a su lugar global
    return np.tril(L)


# =============================================================================
# VALIDACIÓN
# =============================================================================
def residuo_relativo(A, L):
    """||A − L·Lᵀ|| / ||A||  (debe dar ~1e-16: error de redondeo de double)."""
    return np.linalg.norm(A - L @ L.T) / np.linalg.norm(A)


def test(b):
    rank = COMM.Get_rank() if COMM else 0
    p = COMM.Get_size() if COMM else 1
    if rank == 0:
        print("== Validación ==")
        A = np.array([[25, 15, -5], [15, 18, 0], [-5, 0, 11]], dtype=float) # Ejemplo con solución conocida
        esperado = np.array([[5, 0, 0], [3, 3, 0], [-1, 1, 3]], dtype=float)
        ok1 = np.allclose(np.array(cholesky_literal(A.tolist())), esperado)
        ok2 = np.allclose(np.tril(cholesky_optimizado(A.copy(), b=2)), esperado)
        print(f"  3x3 literal:    {'OK' if ok1 else 'FALLA'}")
        print(f"  3x3 optimizado: {'OK' if ok2 else 'FALLA'}")
        n = 300
        A = generar_spd(n)
        ref = np.linalg.cholesky(A)                                         # LAPACK como referencia
        t0 = time.perf_counter()
        L1 = np.array(cholesky_literal(A.tolist()))                         # Medimos cuánto tarda cada versión
        t_lit = time.perf_counter() - t0
        t0 = time.perf_counter()
        L2 = np.tril(cholesky_optimizado(A.copy(), b=64))
        t_opt = time.perf_counter() - t0
        for nombre, L, t in (("literal", L1, t_lit), ("optimizado", L2, t_opt)):
            print(f"  N={n} {nombre:12s} tiempo={t:8.4f} s  residuo={residuo_relativo(A, L):.2e}  "
                  f"max|L-Lref|={np.max(np.abs(L - ref)):.2e}")
    if COMM is None:
        print("  paralelizado: mpi4py no instalado, se omite")
        return
    for n, bb in ((300, 64), (1000, b), (1001, 50)):                        # Incluye N no múltiplo de b (bloque final más chico)
        _, filas = filas_de_proceso(n, bb, rank, p)
        Aloc = generar_filas_spd(n, filas)
        COMM.Barrier()                                                      # Todos arrancan juntos
        t0 = MPI.Wtime()
        cholesky_paralelizado(Aloc, n, bb, COMM)
        COMM.Barrier()                                                      # Esperamos al más lento
        t_par = MPI.Wtime() - t0
        L = juntar_L(Aloc, n, bb, COMM)
        if rank == 0:
            A = generar_spd(n)
            print(f"  N={n} paralelizado ({p} proc, b={bb}) tiempo={t_par:8.4f} s  residuo={residuo_relativo(A, L):.2e}  "
                  f"max|L-Lref|={np.max(np.abs(L - np.linalg.cholesky(A))):.2e}")


# =============================================================================
# BENCHMARKS
# =============================================================================
COLUMNAS = ["version", "N", "p", "b", "reps", "mediana_s", "tiempos_s", "gflops", "residuo"]


def abrir_csv(salida):
    """Abre el CSV para agregar filas. Si existe con otro formato (versión vieja del script),
    lo renombra a *_viejo.csv y arranca uno nuevo, para no mezclar columnas."""
    if os.path.exists(salida):
        with open(salida) as f:
            encabezado = f.readline().strip().split(",")
        if encabezado != COLUMNAS:
            os.replace(salida, salida.replace(".csv", "_viejo.csv"))
    nuevo = not os.path.exists(salida)
    f = open(salida, "a", newline="")
    w = csv.writer(f)
    if nuevo:
        w.writerow(COLUMNAS)
    return f, w


def reps_para(n, reps):
    return reps if reps else (7 if n <= 2000 else (5 if n <= 6000 else 3))  # Menos repeticiones para N grande


def benchmark_serial(tamanos, b, reps, n_max_literal=500, salida="resultados_serial.csv"):
    print("== Benchmark secuencial ==")
    print(f"  {platform.processor() or platform.machine()} | Python {platform.python_version()} | NumPy {np.__version__} | b={b}\n")
    f, w = abrir_csv(salida)
    with f:
        for n in tamanos:
            A = generar_spd(n)
            versiones = [("optimizado", lambda M: cholesky_optimizado(M, b)),
                         ("lapack_ref", np.linalg.cholesky)]                # Referencia profesional ("techo")
            if n <= n_max_literal:
                versiones.insert(0, ("literal", lambda M: cholesky_literal(M.tolist())))
            for nombre, func in versiones:
                r = 1 if nombre == "literal" else reps_para(n, reps)
                ts = []
                for _ in range(r):
                    M = A.copy()                                            # Copia nueva: el optimizado sobrescribe
                    t0 = time.perf_counter()
                    R = func(M)
                    ts.append(time.perf_counter() - t0)
                med = statistics.median(ts)                                 # MEDIANA (lo pide la cátedra)
                res = f"{residuo_relativo(A, np.tril(np.asarray(R))):.2e}" if n <= 3000 else ""
                gf = n**3 / 3 / med / 1e9                                   # Cholesky hace ~n³/3 operaciones
                print(f"  N={n:6d} {nombre:11s} mediana={med:9.3f} s  {gf:6.2f} GFLOP/s  {res}")
                w.writerow([nombre, n, 1, b, r, f"{med:.6f}", ";".join(f"{t:.6f}" for t in ts), f"{gf:.4f}", res])
                f.flush()
            del A
    print(f"\n  Guardado en {salida}")


def benchmark_paralelo(tamanos, b, reps, salida="resultados_paralelo.csv"):
    rank, p = COMM.Get_rank(), COMM.Get_size()
    if rank == 0:
        print(f"== Benchmark paralelo: {p} procesos, b={b} ==\n")
        f, w = abrir_csv(salida)
    for n in tamanos:
        _, filas = filas_de_proceso(n, b, rank, p)
        base = generar_filas_spd(n, filas)                                  # Cada proceso genera SOLO sus filas
        ts = []
        for _ in range(reps_para(n, reps)):
            Aloc = base.copy()
            COMM.Barrier()                                                  # Todos arrancan juntos
            t0 = MPI.Wtime()
            cholesky_paralelizado(Aloc, n, b, COMM)
            COMM.Barrier()                                                  # Esperamos al más lento
            ts.append(MPI.Wtime() - t0)
        res = ""
        if n <= 3000:                                                       # Validación solo para N chico (junta todo en el 0)
            L = juntar_L(Aloc, n, b, COMM)
            if rank == 0:
                res = f"{residuo_relativo(generar_spd(n), L):.2e}"
        if rank == 0:
            med = statistics.median(ts)
            gf = n**3 / 3 / med / 1e9
            print(f"  N={n:6d} paralelizado p={p} mediana={med:9.3f} s  {gf:6.2f} GFLOP/s  {res}")
            w.writerow(["paralelizado", n, p, b, len(ts), f"{med:.6f}", ";".join(f"{t:.6f}" for t in ts), f"{gf:.4f}", res])
            f.flush()
        del base, Aloc
    if rank == 0:
        f.close()
        print(f"\n  Guardado en {salida}")


def tabla_speedup(serial="resultados_serial.csv", paralelo="resultados_paralelo.csv"):
    """Speedup S = T_optimizado / T_paralelo(p)   y   eficiencia E = S / p.
    Se compara contra el MEJOR secuencial (optimizado), no contra el literal."""
    ts = {}
    with open(serial) as f:
        for r in csv.DictReader(f):
            if r["version"] == "optimizado":
                ts[int(r["N"])] = float(r["mediana_s"])                     # Si hay repetidas, queda la última
    tp = {}
    with open(paralelo) as f:
        for r in csv.DictReader(f):
            tp[(int(r["N"]), int(r["p"]))] = float(r["mediana_s"])          # Idem: última medición por (N, p)
    print(f"{'N':>7} {'p':>3} {'T_serial':>10} {'T_paralelo':>11} {'Speedup':>8} {'Eficiencia':>11}")
    for (n, p), t in sorted(tp.items()):
        if n in ts:
            s = ts[n] / t
            print(f"{n:7d} {p:3d} {ts[n]:10.3f} {t:11.3f} {s:8.2f} {s / p:10.0%}")


# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--paralelo", action="store_true")
    ap.add_argument("--speedup", action="store_true")
    ap.add_argument("--N", default="250,500,1000,2000,4000,6000,8000,10000,12000")
    ap.add_argument("--b", type=int, default=128)                           # Tamaño de bloque (probar 64, 128, 256)
    ap.add_argument("--reps", type=int, default=0)                          # 0 = automático según N
    a = ap.parse_args()
    tamanos = [int(x) for x in a.N.split(",")]

    if a.speedup:
        tabla_speedup()
    elif a.paralelo:
        if COMM is None:
            sys.exit("Falta mpi4py. Correr con: mpiexec -n 4 python cholesky.py --paralelo")
        benchmark_paralelo(tamanos, a.b, a.reps)
    elif a.serial:
        benchmark_serial(tamanos, a.b, a.reps)
    else:
        test(a.b)