import numpy as np
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[2]

def ci(y, f):
    y = np.asarray(y); f = np.asarray(f)
    ind = np.argsort(y); y = y[ind]; f = f[ind]
    i = len(y) - 1; j = i - 1; z = 0.0; S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z += 1
                u = f[i] - f[j]
                if u > 0: S += 1
                elif u == 0: S += 0.5
            j -= 1
        i -= 1; j = i - 1
    return S / z if z else float('nan')

ssm_dir = RAIZ / "resultados" / "predicoes" / "ssm_dta"
sub_dir = RAIZ / "resultados" / "predicoes" / "submdta"

rows = {"SSM-DTA": [], "SubMDTA": [], "Ensemble": []}
for fold in range(5):
    ssm_pred = np.array([float(x) for x in open(ssm_dir / f"fold{fold}_pred.txt")])
    true_labels = np.array([float(x) for x in open(RAIZ / "dados" / "davis150_folds" / f"fold{fold}" / "test.label")])
    sub_pred = np.loadtxt(sub_dir / f"fold{fold}_pred.txt")
    ens_pred = (ssm_pred + sub_pred) / 2.0

    for name, preds in [("SSM-DTA", ssm_pred), ("SubMDTA", sub_pred), ("Ensemble", ens_pred)]:
        r = float(np.sqrt(np.mean((true_labels - preds) ** 2)))
        p = float(np.corrcoef(true_labels, preds)[0, 1])
        c = float(ci(true_labels, preds))
        rows[name].append((fold, r, p, c))

for name, vals in rows.items():
    print(f"--- {name} ---")
    for fold, r, p, c in vals:
        print(f"fold {fold}: RMSE={r:.3f} Pearson={p:.3f} CI={c:.3f}")
    arr = np.array([[r, p, c] for _, r, p, c in vals])
    m = arr.mean(axis=0); s = arr.std(axis=0, ddof=0)
    print(f"media: RMSE={m[0]:.3f}+-{s[0]:.3f} Pearson={m[1]:.3f}+-{s[1]:.3f} CI={m[2]:.3f}+-{s[2]:.3f}")
