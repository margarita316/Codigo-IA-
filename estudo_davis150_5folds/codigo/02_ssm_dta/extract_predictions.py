import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default=".")
    ap.add_argument("--python", default="../venv37/Scripts/python.exe")
    ap.add_argument("--data-raw-root", default="../data_raw150")
    ap.add_argument("--data-bin-root", default="../data_bin150")
    ap.add_argument("--ckpt-root", default="../ckpts150")
    ap.add_argument("--out-dir", default="../predictions_ssmdta")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    out_root = repo_dir.parent / Path(args.out_dir).name
    out_root.mkdir(parents=True, exist_ok=True)
    data_bin_abs = (repo_dir / args.data_bin_root).resolve()

    for fold in range(args.folds):
        raw = f"{args.data_raw_root}/fold{fold}"
        ckpt = f"{args.ckpt_root}/fold{fold}/checkpoint_best.pt"
        out_fn = out_root / f"fold{fold}_pred.txt"
        cmd = [
            args.python, "infer.py",
            "--batch-size", "1",
            "--mode", "eval",
            "--checkpoint", ckpt,
            "--data-bin", f"{args.data_bin_root}/fold{fold}",
            "--input-mol-fn", f"{raw}/test.mol.can.re",
            "--input-pro-fn", f"{raw}/test.pro.addspace",
            "--input-label-fn", f"{raw}/test.label",
            "--output-fn", str(out_fn),
        ]
        cmd[cmd.index("--data-bin") + 1] = str(data_bin_abs / f"fold{fold}")
        print("+", " ".join(cmd))
        log_fn = out_root / f"fold{fold}_infer.log"
        with open(log_fn, "w", encoding="utf-8") as logf:
            subprocess.run(cmd, cwd=str(repo_dir), stdout=logf, stderr=subprocess.STDOUT)
        print(f"fold {fold}: predicciones guardadas en {out_fn}")


if __name__ == "__main__":
    main()
