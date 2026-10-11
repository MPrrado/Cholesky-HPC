"""
Gráficos del Cholesky SERIAL — Proyecto 7 (HPC)

Muestra por qué la versión serial no escala:
    1) Tiempo vs N, junto con la curva teórica O(N³)
    2) Velocidad real en GFLOP/s (miles de millones de operaciones por segundo)
    3) Proyección: cuánto tardaría con matrices más grandes

Lee ../resultados/resultados_cholesky_serial{SUFIJO}.csv (el que genera run_serial.sh)
y calcula la mediana de las repeticiones.

Uso:  python graficos_serial.py pc      (o)      python graficos_serial.py cluster
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAQUINA = sys.argv[1] if len(sys.argv) > 1 else "pc"
SUFIJO = "_pc" if MAQUINA == "pc" else ""
AQUI = os.path.dirname(os.path.abspath(__file__))           # carpeta de este script (Cholesky_serial/)
ARCHIVO = os.path.join(AQUI, "..", "resultados", f"resultados_cholesky_serial{SUFIJO}.csv")

AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#8a8984"


def estilo(ax, titulo):
    ax.set_title(titulo, fontsize=11, loc="left")
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=9)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9, frameon=False)


def segundos_a_texto(t):
    """Pasa segundos a algo legible: 45 s, 12 min, 3,5 h, 2 días..."""
    if t < 1:
        return f"{t * 1000:.0f} ms"
    if t < 60:
        return f"{t:.1f} s"
    if t < 3600:
        return f"{t / 60:.0f} min"
    if t < 86400:
        return f"{t / 3600:.1f} h"
    return f"{t / 86400:.1f} días"


# ----------------------------------------------------------------------------- DATOS
df = pd.read_csv(ARCHIVO).dropna(subset=["tiempo"])
med = df.groupby("dimension_matriz").tiempo.median().reset_index().sort_values("dimension_matriz")
N = med.dimension_matriz.to_numpy(dtype=float)
T = med.tiempo.to_numpy()

# Cholesky hace ~N³/3 operaciones -> GFLOP/s = (N³/3) / tiempo / 10⁹
gflops = (N ** 3 / 3) / T / 1e9

# Ajuste T = c·N³ usando las matrices grandes (en las chicas pesa más la lectura del archivo)
grandes = N >= 500 if (N >= 500).sum() >= 2 else N > 0
c = np.median(T[grandes] / N[grandes] ** 3)

print(f"{'N':>7} {'tiempo':>10} {'GFLOP/s':>9}")
for n, t, g in zip(N, T, gflops):
    print(f"{int(n):7d} {segundos_a_texto(t):>10} {g:9.3f}")

# ----------------------------------------------------------------------------- GRÁFICOS
fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle(f"Cholesky serial (1 proceso, sin MPI) — {MAQUINA.upper()} — mediana de las repeticiones",
             fontsize=13, x=0.01, ha="left")

# (1) Tiempo vs N con la curva teórica N³
ax = axs[0]
n_curva = np.logspace(np.log10(N.min()), np.log10(N.max()), 100)
ax.plot(n_curva, c * n_curva ** 3, color=GRIS, linestyle="--", linewidth=1.3, label="Teórico  T ∝ N³")
ax.plot(N, T, color=AZUL, linewidth=2, marker="o", markersize=6, label="Serial (medido)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Dimensión de la matriz (N)"); ax.set_ylabel("Tiempo [s]")
estilo(ax, "Tiempo vs N: duplicar N multiplica el tiempo por ~8")

# (2) GFLOP/s: qué tan bien usa el procesador
ax = axs[1]
ax.plot(N, gflops, color=NARANJA, linewidth=2, marker="o", markersize=6, label="Serial")
ax.set_xscale("log")
ax.set_xlabel("Dimensión de la matriz (N)"); ax.set_ylabel("GFLOP/s")
ax.set_ylim(0, max(gflops) * 1.3)
estilo(ax, "Velocidad real: no aprovecha el procesador")

# (3) Proyección a matrices más grandes con T = c·N³
ax = axs[2]
n_proy = np.array([1000, 2000, 5000, 10000, 20000, 50000])
t_proy = c * n_proy ** 3
barras = ax.bar([f"{n:,}".replace(",", ".") for n in n_proy], t_proy / 3600, color=AZUL, width=0.6)
for barra, t in zip(barras, t_proy):
    ax.annotate(segundos_a_texto(t), (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                ha="center", va="bottom", fontsize=9, color="#0b0b0b", xytext=(0, 2), textcoords="offset points")
ax.set_yscale("log")
ax.set_xlabel("Dimensión de la matriz (N)"); ax.set_ylabel("Tiempo estimado [horas]")
estilo(ax, "Proyección: cuánto tardaría con matrices más grandes")

fig.text(0.01, 0.01, "El tiempo incluye la lectura del archivo .txt. La proyección usa T = c·N³ ajustado con N ≥ 500.",
         fontsize=8.5, color="#52514e")
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
salida = os.path.join(AQUI, f"graficos_serial_{MAQUINA}.png")
fig.savefig(salida, dpi=140)
print(f"\nGuardado en {salida}")
plt.show()
