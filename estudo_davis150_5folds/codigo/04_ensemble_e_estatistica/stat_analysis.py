import json
import numpy as np
from scipy import stats
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[2]

ssm = np.array([0.6363915580649215, 0.7312152072447002, 0.5770557982793489, 1.2505141374477298, 1.1863533880962032])
sub = np.array([0.9426650504025984, 1.3073084709117686, 0.4734350109126208, 1.4200961020523577, 1.488114670693548])
ens = np.array([0.739063040887111, 0.9118464446912771, 0.4146016265917413, 1.2725250859832342, 1.2927628532594642])

pairs = [("SSM-DTA", ssm, "SubMDTA", sub), ("SSM-DTA", ssm, "Ensamble", ens), ("SubMDTA", sub, "Ensamble", ens)]

results = []
for name_a, a, name_b, b in pairs:
    t_stat, p_t = stats.ttest_rel(a, b)
    w_stat, p_w = stats.wilcoxon(a, b)
    diff = a - b
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float('nan')
    row = {
        "comparacion": f"{name_a} vs {name_b}",
        "media_a": float(a.mean()), "media_b": float(b.mean()),
        "diferencia_media": float(diff.mean()),
        "t_pareado": float(t_stat), "p_valor_t": float(p_t),
        "wilcoxon_W": float(w_stat), "p_valor_wilcoxon": float(p_w),
        "cohen_d_pareado": float(d),
        "significativo_0.05": bool(p_t < 0.05),
    }
    results.append(row)
    print(row)

with open(RAIZ / "resultados" / "metricas" / "stat_analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nGuardado en stat_analysis_summary.json")
