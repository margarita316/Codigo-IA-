from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

folds = ["Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4"]
ssm = np.array([0.6363915580649215, 0.7312152072447002, 0.5770557982793489, 1.2505141374477298, 1.1863533880962032])
sub = np.array([0.9426650504025984, 1.3073084709117686, 0.4734350109126208, 1.4200961020523577, 1.488114670693548])
ens = np.array([0.739063040887111, 0.9118464446912771, 0.4146016265917413, 1.2725250859832342, 1.2927628532594642])

means = [ssm.mean(), sub.mean(), ens.mean()]
stds = [ssm.std(ddof=0), sub.std(ddof=0), ens.std(ddof=0)]

COL_SSM = "#2a78d6"
COL_SUB = "#eb6834"
COL_ENS = "#1baf7a"
FOLD_COLORS = ["#c3c2b7", "#9e9c92", "#7a786f", "#565550", "#33322f"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]

fig, ax = plt.subplots(figsize=(8.5, 6), facecolor="white")
ax.set_facecolor(SURFACE)

x = np.array([0, 1, 2])
data = np.vstack([ssm, sub, ens])

for i in range(5):
    ax.plot(x, data[:, i], color=FOLD_COLORS[i], linewidth=1.3, alpha=0.85,
            marker="o", markersize=5, markerfacecolor=FOLD_COLORS[i],
            markeredgecolor="white", markeredgewidth=0.6, zorder=2,
            label=folds[i])

method_colors = [COL_SSM, COL_SUB, COL_ENS]
for xi, m, s, c in zip(x, means, stds, method_colors):
    ax.errorbar(xi, m, yerr=s, fmt="D", color=c, markersize=11,
                markeredgecolor="white", markeredgewidth=1.2,
                elinewidth=2.2, capsize=7, capthick=2.2, zorder=5)
    label = f"{m:.3f}".replace(".", ",") + " \u00b1 " + f"{s:.3f}".replace(".", ",")
    ax.text(xi, m + s + 0.09, label, ha="center", va="bottom",
            fontsize=10, color=INK_PRIMARY, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(["SSM-DTA", "SubMDTA", "Ensemble"], fontsize=12, color=INK_PRIMARY, fontweight="bold")
ax.set_xlim(-0.4, 2.4)
ax.set_ylabel("RMSE", fontsize=11, color=INK_PRIMARY)
ax.set_ylim(0, 1.75)
ax.set_title("RMSE pareado por fold entre os métodos avaliados\n(pontos finos = folds individuais; losangos = média \u00b1 DP)",
              fontsize=12.5, fontweight="bold", color=INK_PRIMARY, pad=16)

ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(BASELINE)
ax.spines["bottom"].set_color(BASELINE)
ax.tick_params(colors=INK_MUTED)

legend = ax.legend(title="Fold", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    fontsize=9.5, labelcolor=INK_PRIMARY, title_fontsize=10)
plt.setp(legend.get_title(), color=INK_PRIMARY)

fig.patch.set_facecolor("white")
plt.tight_layout()
out_path = str(Path(__file__).resolve().parents[2] / "resultados" / "figuras" / "rmse_pareado.png")
plt.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
print("Guardado en", out_path)
