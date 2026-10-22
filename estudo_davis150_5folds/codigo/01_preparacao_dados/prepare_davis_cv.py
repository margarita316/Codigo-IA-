import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import KFold, train_test_split

SMI_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
SMI_REGEX = re.compile(SMI_PATTERN)


def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def tokenize_smiles(smi):
    tokens = SMI_REGEX.findall(smi)
    if re.sub(r"\s+", "", smi) != "".join(tokens):
        return None
    return " ".join(tokens)


def addspace(seq):
    return " ".join(list(seq))


def load_and_clean(data_path):
    df = pd.read_csv(data_path, sep="\t")
    df = df.rename(columns={"ID1": "Drug_ID", "X1": "Drug", "ID2": "Target_ID", "X2": "Target"})
    df = df.dropna(subset=["Drug", "Target", "Y"]).reset_index(drop=True)
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["Y"])
    df = df[df["Y"] > 0].reset_index(drop=True)
    df["pKd"] = 9.0 - np.log10(df["Y"].astype(float))

    print(f"Filas originales: {len(df)}")

    can_cache = {}
    tok_cache = {}
    mols, pros, labels = [], [], []
    dropped = 0
    for drug, target, pkd in zip(df["Drug"], df["Target"], df["pKd"]):
        if drug not in can_cache:
            can_cache[drug] = canonicalize(drug)
        can = can_cache[drug]
        if can is None:
            dropped += 1
            continue
        if can not in tok_cache:
            tok_cache[can] = tokenize_smiles(can)
        tok = tok_cache[can]
        if tok is None:
            dropped += 1
            continue
        mols.append(tok)
        pros.append(addspace(str(target).strip()))
        labels.append(pkd)

    print(f"Pares validos tras canonicalizar/tokenizar: {len(mols)} (descartados: {dropped})")
    return mols, pros, labels


def write_split(out_dir, split_name, mols, pros, labels):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{split_name}.mol.can.re").write_text(
        "\n".join(mols) + "\n", encoding="utf-8"
    )
    (out_dir / f"{split_name}.pro.addspace").write_text(
        "\n".join(pros) + "\n", encoding="utf-8"
    )
    (out_dir / f"{split_name}.label").write_text(
        "\n".join(f"{v:.6f}" for v in labels) + "\n", encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="ruta a davis.tab")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--subsample", type=int, default=None,
                     help="si se indica, toma una muestra aleatoria de este tamano antes de hacer los folds")
    args = ap.parse_args()

    mols, pros, labels = load_and_clean(args.data)

    if args.subsample is not None and args.subsample < len(mols):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(mols), size=args.subsample, replace=False)
        mols = [mols[i] for i in idx]
        pros = [pros[i] for i in idx]
        labels = [labels[i] for i in idx]
        print(f"Submuestra aleatoria: {len(mols)} pares (semilla={args.seed})")
    mols = np.array(mols, dtype=object)
    pros = np.array(pros, dtype=object)
    labels = np.array(labels, dtype=np.float64)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    out_root = Path(args.out_dir)

    for fold, (dev_idx, test_idx) in enumerate(kf.split(mols)):
        train_idx, valid_idx = train_test_split(
            dev_idx, test_size=args.valid_frac, random_state=args.seed + fold
        )
        fold_dir = out_root / f"fold{fold}"
        write_split(fold_dir, "train", mols[train_idx], pros[train_idx], labels[train_idx])
        write_split(fold_dir, "valid", mols[valid_idx], pros[valid_idx], labels[valid_idx])
        write_split(fold_dir, "test", mols[test_idx], pros[test_idx], labels[test_idx])
        print(
            f"fold{fold}: train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}"
        )

    print("Listo.")


if __name__ == "__main__":
    main()
