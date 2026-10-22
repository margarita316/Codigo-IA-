from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

methods = ["SSM-DTA", "SubMDTA"]
params_m = [177.85, 44.47]     # milhoes de parametros
time_min = [95, 8]             # minutos totais (5 folds x 25 epocas)

COL_SSM = "#2a78d6"
COL_SUB = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]

fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="white")

def styled_bar(ax, values, title, ylabel, value_fmt):
    ax.set_facecolor(SURFACE)
    bars = ax.bar([0, 1], values, width=0.5, color=[COL_SSM, COL_SUB], edgecolor="white", linewidth=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.02, value_fmt(v),
                ha="center", va="bottom", fontsize=10.5, color=INK_PRIMARY, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(methods, fontsize=11, color=INK_PRIMARY)
    ax.set_ylabel(ylabel, fontsize=10.5, color=INK_PRIMARY)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK_PRIMARY, pad=12)
    ax.set_ylim(0, max(values) * 1.22)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED)

styled_bar(axes[0], params_m, "Tamanho do modelo", "Parâmetros (milhões)",
           lambda v: f"{v:.1f} M".replace(".", ","))
styled_bar(axes[1], time_min, "Custo computacional total\n(5 folds × 25 épocas)", "Tempo de treinamento (min)",
           lambda v: f"{v:.0f} min")

fig.patch.set_facecolor("white")
plt.tight_layout()
out_path = str(Path(__file__).resolve().parents[2] / "resultados" / "figuras" / "comparacao_modelos.png")
plt.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
print("Guardado en", out_path)
