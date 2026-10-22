"""Orquesta el entrenamiento SSM-DTA baseline (dti_separate, roberta_dti_cross_attn)
con validacion cruzada sobre DAVIS. Por cada fold:
  1) entrena --max-epoch epocas (con checkpoint/resume automatico de fairseq)
  2) evalua checkpoint_best.pt sobre el subset "test" del fold con validate.py
  3) extrae RMSE/MSE del log y las guarda

Al final imprime y guarda un resumen con media +/- desviacion estandar
de las 3 (o N) corridas.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

LOSS_RE = re.compile(r"\|\s*loss\s+([\d.]+)")
RMSE_RE = re.compile(r"loss_regress_rmse\s+([\d.]+)")


def run_logged(cmd, cwd, log_path, append=False):
    print("+", " ".join(str(c) for c in cmd))
    mode = "a" if append else "w"
    with open(log_path, mode, encoding="utf-8") as logf:
        if append:
            logf.write("\n----- reintento -----\n")
        proc = subprocess.run(cmd, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def run_with_retries(cmd, cwd, log_path, max_retries=8):
    rc = run_logged(cmd, cwd, log_path, append=False)
    attempt = 1
    while rc != 0 and attempt <= max_retries:
        print(f"intento {attempt} fallo (rc={rc}), reintentando (retoma desde checkpoint_last.pt si existe)...")
        time.sleep(5)
        rc = run_logged(cmd, cwd, log_path, append=True)
        attempt += 1
    return rc


def parse_test_metrics(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    losses = LOSS_RE.findall(text)
    rmses = RMSE_RE.findall(text)
    loss = float(losses[-1]) if losses else None
    rmse = float(rmses[-1]) if rmses else None
    return loss, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default=".")
    ap.add_argument("--python", default="../venv37/Scripts/python.exe")
    ap.add_argument("--data-bin-root", default="../data_bin3")
    ap.add_argument("--ckpt-root", default="../ckpts3")
    ap.add_argument("--log-root", default="../logs3")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--update-freq", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.00005)
    ap.add_argument("--start-fold", type=int, default=0)
    args = ap.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    py = args.python
    log_root = repo_dir.parent / Path(args.log_root).name
    log_root.mkdir(parents=True, exist_ok=True)
    ckpt_root = repo_dir.parent / Path(args.ckpt_root).name
    ckpt_root.mkdir(parents=True, exist_ok=True)

    results = []
    for fold in range(args.start_fold, args.folds):
        data_bin = f"{args.data_bin_root}/fold{fold}"
        save_dir = f"{args.ckpt_root}/fold{fold}"
        (ckpt_root / f"fold{fold}").mkdir(parents=True, exist_ok=True)

        train_label = repo_dir / data_bin / "label" / "train.label"
        n_train = sum(1 for _ in open(train_label, encoding="utf-8"))
        effective_batch = args.batch_size * args.update_freq
        steps_per_epoch = max(1, n_train // effective_batch)
        total_num_update = steps_per_epoch * args.epochs
        warmup_updates = max(1, min(500, total_num_update // 10))

        train_log = log_root / f"fold{fold}_train.log"
        train_cmd = [
            py, "train.py", "--task", "dti_separate", data_bin,
            "--num-classes", "1", "--init-token", "0",
            "--max-positions-molecule", "512", "--max-positions-protein", "1024",
            "--save-dir", save_dir,
            "--encoder-layers", "12",
            "--criterion", "dti_separate", "--regression-target",
            "--batch-size", str(args.batch_size), "--update-freq", str(args.update_freq),
            "--required-batch-size-multiple", "1",
            "--num-workers", "0",
            "--optimizer", "adam", "--weight-decay", "0.1",
            "--adam-betas", "(0.9,0.98)", "--adam-eps", "1e-06",
            "--lr-scheduler", "polynomial_decay", "--lr", str(args.lr),
            "--warmup-updates", str(warmup_updates), "--max-epoch", str(args.epochs),
            "--total-num-update", str(total_num_update),
            "--clip-norm", "1.0",
            "--arch", "roberta_dti_cross_attn", "--dropout", "0.1", "--attention-dropout", "0.1",
            "--skip-invalid-size-inputs-valid-test",
            "--memory-efficient-fp16",
            "--shorten-method", "truncate",
            "--best-checkpoint-metric", "loss",
            "--no-save-optimizer-state",
            "--no-last-checkpoints",
            "--find-unused-parameters",
            "--log-interval", "50",
        ]
        print(f"\n===== FOLD {fold}: entrenando {args.epochs} epocas =====")
        rc = run_with_retries(train_cmd, cwd=str(repo_dir), log_path=train_log)
        if rc != 0:
            print(f"fold {fold}: entrenamiento fallo tras varios reintentos (ver {train_log}); continuo con el siguiente fold")

        test_log = log_root / f"fold{fold}_test.log"
        ckpt = f"{save_dir}/checkpoint_best.pt"
        infer_cmd = [
            py, "-m", "fairseq_cli.validate",
            "--task", "dti_separate",
            "--batch-size", "8",
            "--num-workers", "0",
            "--valid-subset", "test",
            "--criterion", "dti_separate",
            "--path", ckpt,
            data_bin,
        ]
        print(f"===== FOLD {fold}: evaluando en test =====")
        run_logged(infer_cmd, cwd=str(repo_dir), log_path=test_log)
        loss, rmse = parse_test_metrics(test_log)
        print(f"fold {fold}: test loss(mse)={loss} rmse={rmse}")
        results.append({"fold": fold, "test_mse": loss, "test_rmse": rmse})

        summary_path = repo_dir.parent / "cv_summary.json"
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    rmses = [r["test_rmse"] for r in results if r["test_rmse"] is not None]
    mses = [r["test_mse"] for r in results if r["test_mse"] is not None]
    final = {
        "per_fold": results,
        "rmse_mean": float(np.mean(rmses)) if rmses else None,
        "rmse_std": float(np.std(rmses)) if rmses else None,
        "mse_mean": float(np.mean(mses)) if mses else None,
        "mse_std": float(np.std(mses)) if mses else None,
    }
    (repo_dir.parent / "cv_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("\n=== RESUMEN FINAL ===")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
