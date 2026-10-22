import json

import numpy as np
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[2]  # raiz do estudo (pasta estudo_davis150_5folds)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


ssm_dir = RAIZ / "resultados" / "predicoes" / "ssm_dta"
sub_dir = RAIZ / "resultados" / "predicoes" / "submdta"

rows = []
for fold in range(5):
    ssm_pred = np.array([float(x) for x in open(ssm_dir / f"fold{fold}_pred.txt")])
    true_labels = np.array([float(x) for x in open(RAIZ / "dados" / "davis150_folds" / f"fold{fold}" / "test.label")])

    sub_pred = np.loadtxt(sub_dir / f"fold{fold}_pred.txt")
    sub_true = np.loadtxt(sub_dir / f"fold{fold}_true.txt")

    assert len(ssm_pred) == len(sub_pred) == len(true_labels), (
        f"fold {fold}: tamanos no coinciden ssm={len(ssm_pred)} sub={len(sub_pred)} true={len(true_labels)}"
    )
    assert np.allclose(true_labels, sub_true, atol=1e-3), (
        f"fold {fold}: las etiquetas verdaderas no coinciden entre metodos (orden distinto?)"
    )

    ensemble_pred = (ssm_pred + sub_pred) / 2.0

    row = {
        "fold": fold,
        "rmse_ssmdta": rmse(true_labels, ssm_pred),
        "rmse_submdta": rmse(true_labels, sub_pred),
        "rmse_ensemble": rmse(true_labels, ensemble_pred),
        "pearson_ensemble": pearson(true_labels, ensemble_pred),
    }
    rows.append(row)
    print(row)

rmse_ssm = np.mean([r["rmse_ssmdta"] for r in rows])
rmse_sub = np.mean([r["rmse_submdta"] for r in rows])
rmse_ens = np.mean([r["rmse_ensemble"] for r in rows])
std_ens = np.std([r["rmse_ensemble"] for r in rows])

summary = {
    "per_fold": rows,
    "rmse_mean_ssmdta": float(rmse_ssm),
    "rmse_mean_submdta": float(rmse_sub),
    "rmse_mean_ensemble": float(rmse_ens),
    "rmse_std_ensemble": float(std_ens),
}

out_path = RAIZ / "resultados" / "metricas" / "ensemble_summary.json"
out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\n=== RESUMEN ENSAMBLE ===")
print(json.dumps(summary, indent=2))
