"""
Gráficos de resultados — Proyecto 7: Cholesky paralelo (HPC)

Lee los CSV de medianas de la carpeta resultados/ (los genera obtener_medianas.py):
    resultados_cholesky_txt{SUFIJO}_mediana.csv         Right-looking paralelo, lectura .txt
    resultados_cholesky_bin{SUFIJO}_mediana.csv         Right-looking paralelo, lectura paralela .bin
    resultados_cholesky_scalapack{SUFIJO}_mediana.csv   ScaLAPACK (pdpotrf)
donde SUFIJO es "" para el cluster y "_pc" para la PC.

Ventana 1 (4 gráficos): tiempo vs N con 1 proceso, tiempo vs N con el mejor p,
                        tiempo vs procesos y speedup vs procesos (N = 10000)
Ventana 2 (6 gráficos): speedup según N y eficiencia según N, uno por cada algoritmo
También guarda cada ventana como .png y cada gráfico por separado en imagenes_<maquina>/ (para el README).

Uso:  python graficos.py cluster      (o)      python graficos.py pc
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

# ----------------------------------------------------------------------------- CONFIGURACIÓN
MAQUINA = sys.argv[1] if len(sys.argv) > 1 else "cluster"
SUFIJO = "_pc" if MAQUINA == "pc" else ""
# Rutas armadas desde la carpeta de este script (graficar/): funciona sin importar desde dónde se ejecute
AQUI = os.path.dirname(os.path.abspath(__file__))
CARPETA = os.path.join(AQUI, "..", "resultados")
CARPETA_IMAGENES = os.path.join(AQUI, f"imagenes_{MAQUINA}")

ALGORITMOS = {                               # nombre en el gráfico: (archivo, color)
    "Right-looking (.txt)": (f"resultados_cholesky_txt{SUFIJO}_mediana.csv",       "#eb6834"),
    "Right-looking (.bin)": (f"resultados_cholesky_bin{SUFIJO}_mediana.csv",       "#1baf7a"),
    "ScaLAPACK (pdpotrf)":  (f"resultados_cholesky_scalapack{SUFIJO}_mediana.csv", "#4a3aa7"),
}
N_COMPARAR = 10000                                       # N usado para comparar los algoritmos
N_SPEEDUP = [500, 1000, 2000, 5000, 10000]               # N que se dibujan en los gráficos "según N"
TONOS_N = ["#e8a317", "#e0592a", "#c2185b", "#6a1b9a", "#1a237e"]   # amarillo -> naranja -> rojo -> violeta -> azul
MARCAS_N = ["o", "s", "^", "D", "v"]                                # además del color, cada N tiene su forma de punto
GRIS = "#8a8984"
NOTA = ("Nota: mediana de las repeticiones. Los tiempos del right-looking incluyen la lectura del archivo; "
        "ScaLAPACK mide solo la factorización. Speedup calculado contra el mismo algoritmo con 1 proceso.")


# ----------------------------------------------------------------------------- DATOS
def cargar():
    """Devuelve {nombre: DataFrame} solo con los CSV que existen."""
    datos = {}
    for nombre, (archivo, _) in ALGORITMOS.items():
        ruta = os.path.join(CARPETA, archivo)
        if os.path.exists(ruta):
            datos[nombre] = pd.read_csv(ruta)
        else:
            print(f"Aviso: no se encontró {ruta}, se omite '{nombre}'.")
    return datos


def speedup(df, n):
    """Speedup S(p) = T(1)/T(p) y eficiencia E(p) = S(p)/p para un N.
    Si no hay medición con 1 proceso, devuelve listas vacías."""
    sub = df[df.dimension_matriz == n].sort_values("numero_de_nucleos")
    t1 = sub[sub.numero_de_nucleos == 1].tiempo
    if sub.empty or t1.empty:
        return [], [], []
    p = sub.numero_de_nucleos.tolist()
    s = (float(t1.iloc[0]) / sub.tiempo).tolist()
    return p, s, [si / pi for si, pi in zip(s, p)]


# ----------------------------------------------------------------------------- ESTILO
def eje_procesos(ax, procesos):
    """Eje x en escala log2 con marcas en 1, 2, 4, 8, ..."""
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(procesos))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(p) for p in procesos])
    ax.set_xlim(procesos[0] * 0.85, procesos[-1] * 1.15)   # misma escala en todos los gráficos
    ax.set_xlabel("Procesos (p)")


def eje_speedup(ax):
    """Eje y en escala log2 pero con números comunes (0.5, 1, 2, 4...)."""
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("Speedup  S = T(1) / T(p)")


def eje_eficiencia(ax, tope):
    ax.set_ylabel("Eficiencia  E = S / p")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, max(1.15, tope * 1.1))               # la eficiencia puede pasar el 100 % (efecto caché)


def estilo(ax, titulo):
    ax.set_title(titulo, fontsize=11, loc="left")
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=9)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8.5, frameon=False)


def linea(ax, x, y, color, etiqueta, marca="o"):
    ax.plot(x, y, color=color, linewidth=2, marker=marca, markersize=6, label=etiqueta)


def color(nombre):
    return ALGORITMOS[nombre][1]


# ----------------------------------------------------------------------------- GRÁFICOS
def tiempo_vs_n_1_proceso(ax, datos, procesos):
    for nombre, df in datos.items():
        sub = df[df.numero_de_nucleos == 1].sort_values("dimension_matriz")
        if not sub.empty:
            linea(ax, sub.dimension_matriz, sub.tiempo, color(nombre), nombre)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Dimensión de la matriz (N)"); ax.set_ylabel("Tiempo [s]")
    estilo(ax, "Tiempo vs N con 1 proceso")


def tiempo_vs_n_mejor_p(ax, datos, procesos):
    for nombre, df in datos.items():
        mejor = df.loc[df.groupby("dimension_matriz").tiempo.idxmin()].sort_values("dimension_matriz")
        linea(ax, mejor.dimension_matriz, mejor.tiempo, color(nombre), nombre)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Dimensión de la matriz (N)"); ax.set_ylabel("Tiempo [s]")
    estilo(ax, "Tiempo vs N con el mejor p de cada uno")


def tiempo_vs_procesos(ax, datos, procesos):
    for nombre, df in datos.items():
        sub = df[df.dimension_matriz == N_COMPARAR].sort_values("numero_de_nucleos")
        if not sub.empty:
            linea(ax, sub.numero_de_nucleos, sub.tiempo, color(nombre), nombre)
    ax.set_yscale("log"); eje_procesos(ax, procesos)
    ax.set_ylabel("Tiempo [s]")
    estilo(ax, f"Tiempo vs procesos (N = {N_COMPARAR})")


def speedup_vs_procesos(ax, datos, procesos):
    ax.plot(procesos, procesos, color=GRIS, linestyle="--", linewidth=1.2, label="Ideal (S = p)")
    for nombre, df in datos.items():
        p, s, _ = speedup(df, N_COMPARAR)
        if p:
            linea(ax, p, s, color(nombre), nombre)
    eje_speedup(ax); eje_procesos(ax, procesos)
    estilo(ax, f"Speedup vs procesos (N = {N_COMPARAR})")


def speedup_segun_n(ax, datos, procesos, nombre):
    ax.plot(procesos, procesos, color=GRIS, linestyle="--", linewidth=1.2, label="Ideal")
    if nombre in datos:
        for n, tono, marca in zip(N_SPEEDUP, TONOS_N, MARCAS_N):
            p, s, _ = speedup(datos[nombre], n)
            if p:
                linea(ax, p, s, tono, f"N = {n}", marca)
    eje_speedup(ax); eje_procesos(ax, procesos)
    estilo(ax, f"Speedup según N — {nombre}")


def eficiencia_segun_n(ax, datos, procesos, nombre):
    ax.axhline(1.0, color=GRIS, linestyle="--", linewidth=1.2, label="Ideal (100 %)")
    tope = 1.0
    if nombre in datos:
        for n, tono, marca in zip(N_SPEEDUP, TONOS_N, MARCAS_N):
            p, _, e = speedup(datos[nombre], n)
            if p:
                linea(ax, p, e, tono, f"N = {n}", marca)
                tope = max(tope, max(e))
    eje_procesos(ax, procesos); eje_eficiencia(ax, tope)
    estilo(ax, f"Eficiencia según N — {nombre}")


# Atajos para usar los gráficos "según N" de cada algoritmo en las listas de abajo
def speedup_n_txt(ax, d, p):          speedup_segun_n(ax, d, p, "Right-looking (.txt)")
def speedup_n_bin(ax, d, p):          speedup_segun_n(ax, d, p, "Right-looking (.bin)")
def speedup_n_scalapack(ax, d, p):    speedup_segun_n(ax, d, p, "ScaLAPACK (pdpotrf)")
def eficiencia_n_txt(ax, d, p):       eficiencia_segun_n(ax, d, p, "Right-looking (.txt)")
def eficiencia_n_bin(ax, d, p):       eficiencia_segun_n(ax, d, p, "Right-looking (.bin)")
def eficiencia_n_scalapack(ax, d, p): eficiencia_segun_n(ax, d, p, "ScaLAPACK (pdpotrf)")


# Ventana 1 (2 x 2): comparación de los tres algoritmos
VENTANA_1 = [tiempo_vs_n_1_proceso, tiempo_vs_n_mejor_p, tiempo_vs_procesos, speedup_vs_procesos]
# Ventana 2 (2 x 3): arriba speedup según N, abajo eficiencia según N, una columna por algoritmo
VENTANA_2 = [speedup_n_txt, speedup_n_bin, speedup_n_scalapack,
             eficiencia_n_txt, eficiencia_n_bin, eficiencia_n_scalapack]


# ----------------------------------------------------------------------------- ARMADO
def ventana(funciones, filas, columnas, tamanio, titulo, archivo, datos, procesos):
    fig, axs = plt.subplots(filas, columnas, figsize=tamanio)
    fig.suptitle(titulo, fontsize=14, x=0.01, ha="left")
    for ax, funcion in zip(axs.flat, funciones):
        funcion(ax, datos, procesos)
    fig.text(0.01, 0.005, NOTA, fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(os.path.join(AQUI, archivo), dpi=130)
    print(f"Guardado {archivo}")


def graficar(datos):
    procesos = sorted({int(p) for df in datos.values() for p in df.numero_de_nucleos})

    # Cada gráfico por separado (para el README)
    os.makedirs(CARPETA_IMAGENES, exist_ok=True)
    for i, funcion in enumerate(VENTANA_1 + VENTANA_2, start=1):
        fig, ax = plt.subplots(figsize=(7, 4.6))
        funcion(ax, datos, procesos)
        fig.tight_layout()
        fig.savefig(os.path.join(CARPETA_IMAGENES, f"{i}_{funcion.__name__}.png"), dpi=150)
        plt.close(fig)
    print(f"Gráficos individuales guardados en {CARPETA_IMAGENES}/")

    # Las dos ventanas
    ventana(VENTANA_1, 2, 2, (15, 10), f"Tiempos y speedup de Cholesky paralelo — {MAQUINA.upper()} (mediana de las repeticiones)",
            f"comparacion_{MAQUINA}.png", datos, procesos)
    ventana(VENTANA_2, 2, 3, (19, 10), f"Speedup y eficiencia según N — {MAQUINA.upper()} (mediana de las repeticiones)",
            f"speedup_eficiencia_{MAQUINA}.png", datos, procesos)
    plt.show()


if __name__ == "__main__":
    datos = cargar()
    if not datos:
        print("No hay CSV de medianas. Corré primero obtener_medianas.py dentro de resultados/")
    else:
        graficar(datos)