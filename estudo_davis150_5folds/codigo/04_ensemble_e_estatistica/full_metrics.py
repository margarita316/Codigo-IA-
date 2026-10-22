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

ssm_rmse, ssm_pearson, ssm_ci = [], [], []
sub_rmse, sub_pearson, sub_ci = [], [], []
ens_rmse, ens_pearson, ens_ci = [], [], []

for fold in range(5):
    ssm_pred = np.array([float(x) for x in open(ssm_dir / f"fold{fold}_pred.txt")])
    true_labels = np.array([float(x) for x in open(RAIZ / "dados" / "davis150_folds" / f"fold{fold}" / "test.label")])
    sub_pred = np.loadtxt(sub_dir / f"fold{fold}_pred.txt")

    ens_pred = (ssm_pred + sub_pred) / 2.0

    for preds, rmse_l, pear_l, ci_l in [
        (ssm_pred, ssm_rmse, ssm_pearson, ssm_ci),
        (sub_pred, sub_rmse, sub_pearson, sub_ci),
        (ens_pred, ens_rmse, ens_pearson, ens_ci),
    ]:
        rmse_l.append(float(np.sqrt(np.mean((true_labels - preds) ** 2))))
        pear_l.append(float(np.corrcoef(true_labels, preds)[0, 1]))
        ci_l.append(float(ci(true_labels, preds)))

def fmt(arr):
    a = np.array(arr)
    return f"{a.mean():.3f} \u00b1 {a.std(ddof=0):.3f}".replace(".", ",")

print("Método       | RMSE            | Pearson         | C-index")
print("SSM-DTA      |", fmt(ssm_rmse), "|", fmt(ssm_pearson), "|", fmt(ssm_ci))
print("SubMDTA      |", fmt(sub_rmse), "|", fmt(sub_pearson), "|", fmt(sub_ci))
print("Ensemble     |", fmt(ens_rmse), "|", fmt(ens_pearson), "|", fmt(ens_ci))
