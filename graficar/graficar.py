import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

# -------------------------------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------------------------------
MAQUINA = sys.argv[1] if len(sys.argv) > 1 else "cluster"
SUFIJO = "_pc" if MAQUINA == "pc" else ""

DIR_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resultados")
FILE_SERIAL = os.path.join(DIR_BASE, f"resultados_cholesky_serial{SUFIJO}.csv")

ALGORITMOS = {
    "TXT": {"ruta": os.path.join(DIR_BASE, f"resultados_cholesky_txt{SUFIJO}_mediana.csv"), "color": "#eb6834", "titulo": "Right-looking (.txt)"},
    "BIN": {"ruta": os.path.join(DIR_BASE, f"resultados_cholesky_bin{SUFIJO}_mediana.csv"), "color": "#1baf7a", "titulo": "Right-looking (.bin)"},
    "ScaLAPACK": {"ruta": os.path.join(DIR_BASE, f"resultados_cholesky_scalapack{SUFIJO}_mediana.csv"), "color": "#4a3aa7", "titulo": "ScaLAPACK (pdpotrf)"}
}

N_REF = 10000
P_MAX = 96

# Configuración específica para el gráfico 2x3
N_PLOT = [500, 1000, 2000, 5000, 10000]
TONOS_N = ["#e8a317", "#e0592a", "#c2185b", "#6a1b9a", "#1a237e"]
MARCAS_N = ["o", "s", "^", "D", "v"]

# -------------------------------------------------------------------------
# UTILIDADES
# -------------------------------------------------------------------------
def estilo(ax, titulo, xlog=False, ylog=False, procesos=None):
    ax.set_title(titulo, fontsize=11, loc="left", pad=10)
    ax.grid(True, color="#e4e3df", linewidth=0.8, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    
    if xlog: 
        ax.set_xscale("log", base=2 if xlog == 2 else 10)
    if ylog: 
        ax.set_yscale("log", base=2 if ylog == 2 else 10)
        
    if procesos is not None:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(procesos))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([str(p) for p in procesos])
        ax.set_xlim(min(procesos) * 0.85, max(procesos) * 1.15)
        
    if ax.get_legend_handles_labels()[0]: 
        ax.legend(frameon=False, fontsize=9)

def format_t(t):
    if t < 1: return f"{t*1000:.0f} ms"
    if t < 60: return f"{t:.1f} s"
    if t < 3600: return f"{t/60:.0f} m"
    return f"{t/3600:.1f} h"

def cargar_datos():
    dfs = {}
    for clave, config in ALGORITMOS.items():
        if os.path.exists(config["ruta"]):
            dfs[clave] = {"df": pd.read_csv(config["ruta"]), "color": config["color"], "titulo": config["titulo"]}
    return dfs

def calcular_speedup(df, n):
    df_n = df[df["dimension_matriz"] == n].sort_values("numero_de_nucleos")
    if df_n.empty or 1 not in df_n["numero_de_nucleos"].values:
        return [], [], []
    t1 = df_n[df_n["numero_de_nucleos"] == 1]["tiempo"].iloc[0]
    p = df_n["numero_de_nucleos"].values
    t = df_n["tiempo"].values
    s = t1 / t
    e = s / p
    return p, s, e

# -------------------------------------------------------------------------
# GRÁFICO 1: PARALELISMO GENERAL (4 subplots)
# -------------------------------------------------------------------------
def graficar_paralelo(dfs, procesos):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Rendimiento Cholesky Paralelo - {MAQUINA.upper()}", fontsize=14, ha="left", x=0.05)
    ax_sp, ax_ef, ax_t96, ax_t1 = axs.flatten()

    for nom, data in dfs.items():
        df, c, titulo = data["df"], data["color"], data["titulo"]

        p, s, e = calcular_speedup(df, N_REF)
        if len(p) > 0:
            ax_sp.plot(p, s, marker="o", color=c, label=titulo, zorder=3)
            ax_ef.plot(p, e, marker="o", color=c, label=titulo, zorder=3)

        for p_target, ax in [(P_MAX, ax_t96), (1, ax_t1)]:
            df_p = df[df["numero_de_nucleos"] == p_target].sort_values("dimension_matriz")
            if not df_p.empty:
                ax.plot(df_p["dimension_matriz"], df_p["tiempo"], marker="o", color=c, label=titulo, zorder=3)

    ax_sp.plot([1, P_MAX], [1, P_MAX], "k--", alpha=0.5, label="Ideal")
    ax_ef.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Ideal")
    
    estilo(ax_sp, f"Speedup (N={N_REF})", ylog=2, procesos=procesos)
    estilo(ax_ef, f"Eficiencia (N={N_REF})", procesos=procesos)
    estilo(ax_t96, f"Tiempo vs N (P={P_MAX})", xlog=10, ylog=10)
    estilo(ax_t1, "Tiempo vs N (P=1)", xlog=10, ylog=10)

    ax_sp.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_ef.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

    for ax in [ax_sp, ax_ef]: ax.set_xlabel("Procesos")
    for ax in [ax_t96, ax_t1]: ax.set_xlabel("Dimensión (N)"); ax.set_ylabel("Tiempo [s]")
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# -------------------------------------------------------------------------
# GRÁFICO 2: SPEEDUP Y EFICIENCIA SEGÚN N (2x3 subplots)
# -------------------------------------------------------------------------
def graficar_segun_n(dfs, procesos):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Speedup y Eficiencia según N - {MAQUINA.upper()}", fontsize=14, ha="left", x=0.01)
    
    orden_cols = ["TXT", "BIN", "ScaLAPACK"]
    
    for i, clave in enumerate(orden_cols):
        ax_sp = axs[0, i]
        ax_ef = axs[1, i]
        
        ax_sp.plot(procesos, procesos, color="gray", linestyle="--", linewidth=1.2, label="Ideal")
        ax_ef.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="Ideal (100%)")
        
        if clave in dfs:
            df = dfs[clave]["df"]
            titulo = dfs[clave]["titulo"]
            
            for n, tono, marca in zip(N_PLOT, TONOS_N, MARCAS_N):
                p, s, e = calcular_speedup(df, n)
                if len(p) > 0:
                    ax_sp.plot(p, s, color=tono, marker=marca, label=f"N = {n}", linewidth=2, zorder=3)
                    ax_ef.plot(p, e, color=tono, marker=marca, label=f"N = {n}", linewidth=2, zorder=3)
                    
            estilo(ax_sp, f"Speedup según N — {titulo}", ylog=2, procesos=procesos)
            estilo(ax_ef, f"Eficiencia según N — {titulo}", procesos=procesos)
            
        else:
            estilo(ax_sp, f"Speedup según N — {clave} (Sin datos)", procesos=procesos)
            estilo(ax_ef, f"Eficiencia según N — {clave} (Sin datos)", procesos=procesos)

        ax_sp.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax_ef.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        
        ax_sp.set_ylabel("Speedup S = T(1) / T(p)")
        ax_ef.set_ylabel("Eficiencia E = S / p")
        ax_sp.set_xlabel("Procesos (p)")
        ax_ef.set_xlabel("Procesos (p)")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# -------------------------------------------------------------------------
# GRÁFICO 3: ANÁLISIS SERIAL (2 subplots)
# -------------------------------------------------------------------------
def graficar_serial():
    if not os.path.exists(FILE_SERIAL):
        return None

    df = pd.read_csv(FILE_SERIAL).dropna(subset=["tiempo"])
    med = df.groupby("dimension_matriz")["tiempo"].median().reset_index().sort_values("dimension_matriz")
    N = med["dimension_matriz"].values
    T = med["tiempo"].values

    idx_grandes = N >= 500 if (N >= 500).sum() >= 2 else N > 0
    c = np.median(T[idx_grandes] / N[idx_grandes]**3)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Análisis Cholesky Serial - {MAQUINA.upper()}", fontsize=14, ha="left", x=0.05)
    ax_curva, ax_barras = axs

    n_teo = np.logspace(np.log10(N.min()), np.log10(N.max()), 50)
    ax_curva.plot(n_teo, c * n_teo**3, "--", color="gray", label="Teórico O(N³)")
    ax_curva.plot(N, T, marker="o", color="#2a78d6", linewidth=2, label="Medido")
    estilo(ax_curva, "Tiempo vs Dimensión (Crecimiento)", xlog=10, ylog=10)
    ax_curva.set_xlabel("Dimensión (N)"); ax_curva.set_ylabel("Tiempo [s]")

    n_proy = np.array([1000, 5000, 10000, 20000, 50000])
    t_proy = c * n_proy**3
    
    barras = ax_barras.bar([str(n) for n in n_proy], t_proy, color="#2a78d6", zorder=3)
    estilo(ax_barras, "Proyección de Tiempo hasta N=50k", ylog=10)
    ax_barras.set_xlabel("Dimensión (N)")
    ax_barras.set_ylabel("Tiempo Estimado [s]")

    for barra, t in zip(barras, t_proy):
        ax_barras.text(barra.get_x() + barra.get_width()/2, barra.get_height() * 1.1,
                       format_t(t), ha='center', va='bottom', fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# -------------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# -------------------------------------------------------------------------
if __name__ == "__main__":
    datos_paralelo = cargar_datos()
    
    if datos_paralelo:
        # Obtenemos todos los núcleos usados para armar el eje X correctamente
        procesos_unicos = sorted({int(p) for d in datos_paralelo.values() for p in d["df"]["numero_de_nucleos"]})
        
        f1 = graficar_paralelo(datos_paralelo, procesos_unicos)
        f1.savefig(os.path.join(DIR_BASE, "..", "graficar", f"grafico_multiples_{MAQUINA}.png"), dpi=120)

        f2 = graficar_segun_n(datos_paralelo, procesos_unicos)
        f2.savefig(os.path.join(DIR_BASE, "..", "graficar", f"speedup_eficiencia_{MAQUINA}.png"), dpi=120)
    else:
        print("No se encontraron CSVs de algoritmos paralelos.")

    f3 = graficar_serial()
    if f3:
        f3.savefig(os.path.join(DIR_BASE, "..", "graficar", f"grafico_serial_{MAQUINA}.png"), dpi=120)

    plt.show()