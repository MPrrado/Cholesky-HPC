"""
Gráficos de resultados — Proyecto 7: Cholesky paralelo (HPC)

Lee los CSV de medianas que genera obtener_medianas.py:
    resultados_cholesky_txt_mediana.csv     (C, lectura .txt por el proceso 0)
    resultados_cholesky_bin_mediana.csv     (C, lectura paralela .bin)
    resultados_cholesky_python_mediana.csv  (Python, NumPy por bloques + mpi4py)
y arma TODOS los gráficos en una sola ventana. También la guarda como graficos.png.

Uso:  python graficos.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

# ----------------------------------------------------------------------------- CONFIGURACIÓN
IMPLEMENTACIONES = {                                     # nombre en el gráfico: (archivo, color)
    "C - lectura .txt":     ("resultados_cholesky_txt_mediana.csv",    "#2a78d6"),
    "C - lectura .bin":     ("resultados_cholesky_bin_mediana.csv",    "#eb6834"),
    "Python (bloques)":     ("resultados_cholesky_python_mediana.csv", "#1baf7a"),
}
N_COMPARAR = 5000                                        # N usado para comparar implementaciones
N_SPEEDUP = [500, 1000, 2000, 5000]                      # N que se dibujan en speedup / eficiencia
TONOS_N = ["#9ec5f4", "#5b9ce8", "#2a78d6", "#0f4a91"]   # Un solo color, de claro (N chico) a oscuro (N grande)
GRIS = "#8a8984"


# ----------------------------------------------------------------------------- LECTURA
def cargar():
    """Devuelve {nombre: DataFrame} solo con los CSV que existen."""
    datos = {}
    for nombre, (archivo, _) in IMPLEMENTACIONES.items():
        if os.path.exists(archivo):
            datos[nombre] = pd.read_csv(archivo)
        else:
            print(f"Aviso: no se encontró {archivo}, se omite '{nombre}'.")
    return datos


def tiempo(df, n, p):
    """Mediana para (N, procesos) o None si no se midió."""
    fila = df[(df.dimension_matriz == n) & (df.numero_de_nucleos == p)]
    return float(fila.tiempo.iloc[0]) if len(fila) else None


def speedup(df, n):
    """Speedup S(p) = T(1) / T(p) y eficiencia E(p) = S(p) / p para un N.
    Si no hay medición con 1 proceso para ese N, devuelve listas vacías."""
    t1 = tiempo(df, n, 1)
    sub = df[df.dimension_matriz == n].sort_values("numero_de_nucleos")
    if t1 is None or sub.empty:
        return [], [], []
    p = sub.numero_de_nucleos.tolist()
    s = [t1 / t for t in sub.tiempo]
    e = [si / pi for si, pi in zip(s, p)]
    return p, s, e


# ----------------------------------------------------------------------------- ESTILO
def eje_procesos(ax, procesos):
    """Eje x en escala log2 con las marcas en 1, 2, 4, 8, ... (los p medidos)."""
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(procesos))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(p) for p in procesos])
    ax.set_xlabel("Procesos (p)")


def eje_speedup(ax):
    """Eje y en escala log2 pero con números comunes (0.5, 1, 2, 4, 8...) en vez de 2^n."""
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("Speedup  S = T(1) / T(p)")


def estilo(ax, titulo):
    ax.set_title(titulo, fontsize=11, loc="left", color="#0b0b0b")
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#b5b4ae")
    ax.tick_params(colors="#52514e", labelsize=9)


def linea(ax, x, y, color, etiqueta):
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5, label=etiqueta)


# ----------------------------------------------------------------------------- GRÁFICOS
def graficar(datos):
    procesos = sorted({p for df in datos.values() for p in df.numero_de_nucleos})
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle("Factorización de Cholesky paralela — resultados (mediana de las repeticiones)",
                 fontsize=14, x=0.01, ha="left")

    # (1) Tiempo vs N con 1 proceso: costo base de cada implementación
    ax = axs[0, 0]
    for nombre, df in datos.items():
        sub = df[df.numero_de_nucleos == 1].sort_values("dimension_matriz")
        if not sub.empty:
            linea(ax, sub.dimension_matriz, sub.tiempo, IMPLEMENTACIONES[nombre][1], nombre)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Dimensión de la matriz (N)"); ax.set_ylabel("Tiempo [s]")
    estilo(ax, "1) Tiempo vs N con 1 proceso")
    ax.legend(fontsize=9, frameon=False)

    # (2) Tiempo vs N con el MEJOR p de cada implementación
    ax = axs[0, 1]
    for nombre, df in datos.items():
        mejor = df.loc[df.groupby("dimension_matriz").tiempo.idxmin()].sort_values("dimension_matriz")
        linea(ax, mejor.dimension_matriz, mejor.tiempo, IMPLEMENTACIONES[nombre][1], nombre)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Dimensión de la matriz (N)"); ax.set_ylabel("Tiempo [s]")
    estilo(ax, "2) Tiempo vs N con el mejor p de cada una")
    ax.legend(fontsize=9, frameon=False)

    # (3) Tiempo vs procesos para un N fijo: las tres implementaciones juntas
    ax = axs[0, 2]
    for nombre, df in datos.items():
        sub = df[df.dimension_matriz == N_COMPARAR].sort_values("numero_de_nucleos")
        if not sub.empty:
            linea(ax, sub.numero_de_nucleos, sub.tiempo, IMPLEMENTACIONES[nombre][1], nombre)
    ax.set_yscale("log"); eje_procesos(ax, procesos)
    ax.set_ylabel("Tiempo [s]")
    estilo(ax, f"3) Tiempo vs procesos (N = {N_COMPARAR})")
    ax.legend(fontsize=9, frameon=False)

    # (4) Speedup vs procesos para un N fijo: las tres implementaciones juntas
    ax = axs[1, 0]
    ax.plot(procesos, procesos, color=GRIS, linestyle="--", linewidth=1.2, label="Ideal (S = p)")
    for nombre, df in datos.items():
        p, s, _ = speedup(df, N_COMPARAR)
        if p:
            linea(ax, p, s, IMPLEMENTACIONES[nombre][1], nombre)
    eje_speedup(ax); eje_procesos(ax, procesos)
    estilo(ax, f"4) Speedup vs procesos (N = {N_COMPARAR})")
    ax.legend(fontsize=9, frameon=False)

    # (5) y (6) Speedup y eficiencia de la versión .bin para varios N
    base = "C - lectura .bin" if "C - lectura .bin" in datos else next(iter(datos))
    df = datos[base]
    ax_s, ax_e = axs[1, 1], axs[1, 2]
    ax_s.plot(procesos, procesos, color=GRIS, linestyle="--", linewidth=1.2, label="Ideal")
    ax_e.axhline(1.0, color=GRIS, linestyle="--", linewidth=1.2, label="Ideal (100 %)")
    for n, color in zip(N_SPEEDUP, TONOS_N):
        p, s, e = speedup(df, n)
        if p:
            linea(ax_s, p, s, color, f"N = {n}")
            linea(ax_e, p, e, color, f"N = {n}")
    eje_speedup(ax_s); eje_procesos(ax_s, procesos)
    estilo(ax_s, f"5) Speedup según N — {base}")
    ax_s.legend(fontsize=9, frameon=False)
    eje_procesos(ax_e, procesos)
    ax_e.set_ylabel("Eficiencia  E = S / p")
    ax_e.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_e.set_ylim(0, 1.1)
    estilo(ax_e, f"6) Eficiencia según N — {base}")
    ax_e.legend(fontsize=9, frameon=False)

    fig.text(0.01, 0.005,
             "Nota: los tiempos de C incluyen la lectura del archivo; los de Python no (genera la matriz en memoria). "
             "Speedup calculado contra la misma implementación con 1 proceso.",
             fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig("graficos.png", dpi=150)
    print("Guardado en graficos.png")
    plt.show()


if __name__ == "__main__":
    datos = cargar()
    if not datos:
        print("No hay CSV de medianas. Corré primero obtener_medianas.py")
    else:
        graficar(datos)