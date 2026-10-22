"""SubMDTA con 5-fold CV sobre la misma submuestra de 150 pares de DAVIS
(reutiliza los mismos splits ya generados para SSM-DTA en data_raw150,
para que ambos metodos sean comparables/ensemblables par a par).

Arquitectura SIN MODIFICAR: GINConvNet + GraphEnhance (sub_50000.pth
preentrenado), tal cual estan en models/ginconv.py y sub.py.
Solo se adapta el pipeline de datos (create_data.py) para nuestra CV
y se reduce NUM_EPOCHS de 1200 a 25 (pedido explicito del usuario).
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from scipy import stats
from torch_geometric.data import DataLoader

from sub import GraphEnhance
from models.ginconv import GINConvNet
from utils import TestbedDataset, rmse, mse, pearson, spearman, ci, get_rm2

import re


# ---- funciones copiadas de create_data.py (pipeline de datos, sin tocar el modelo) ----

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    import networkx as nx
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index


def make_split_sequence(word_dict_2, word_dict_3, word_dict_4):
    def split_sequence(sequence, ngram):
        sequence = '-' + sequence + '='
        d = {2: word_dict_2, 3: word_dict_3, 4: word_dict_4}[ngram]
        return [d[sequence[i:i + ngram]] for i in range(len(sequence) - ngram + 1)]
    return split_sequence


def fun(prots, ngram, split_sequence):
    XT = [split_sequence(t, ngram=ngram) for t in prots]
    seq_max = 1200
    proteins = []
    for pro in XT:
        pro_len = len(pro)
        if pro_len > seq_max:
            pro = pro[:seq_max]
        else:
            pro = pro + [0] * (seq_max - pro_len)
        proteins.append(pro)
    return proteins


def read_fold_split(raw_dir, split):
    mol_lines = [l.strip().replace(' ', '') for l in open(f"{raw_dir}/{split}.mol.can.re", encoding='utf-8')]
    pro_lines = [l.strip().replace(' ', '') for l in open(f"{raw_dir}/{split}.pro.addspace", encoding='utf-8')]
    labels = [float(l.strip()) for l in open(f"{raw_dir}/{split}.label", encoding='utf-8')]
    return mol_lines, pro_lines, labels


def train_one_epoch(model, device, loader, optimizer, loss_fn):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()


def predict(model, device, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            preds.append(output.cpu().numpy().flatten())
            labels.append(data.y.view(-1).cpu().numpy())
    return np.concatenate(labels), np.concatenate(preds)


def run_fold(fold, args, device):
    raw_dir = Path(args.data_raw_root) / f"fold{fold}"
    train_mol, train_pro, train_y = read_fold_split(raw_dir, "train")
    valid_mol, valid_pro, valid_y = read_fold_split(raw_dir, "valid")
    test_mol, test_pro, test_y = read_fold_split(raw_dir, "test")

    word_dict_2 = defaultdict(lambda: len(word_dict_2))
    word_dict_3 = defaultdict(lambda: len(word_dict_3))
    word_dict_4 = defaultdict(lambda: len(word_dict_4))
    split_sequence = make_split_sequence(word_dict_2, word_dict_3, word_dict_4)

    all_mol = list(dict.fromkeys(train_mol + valid_mol + test_mol))
    smile_graph = {s: smile_to_graph(s) for s in all_mol}

    proc_root = Path(args.proc_root) / f"fold{fold}"
    proc_root.mkdir(parents=True, exist_ok=True)

    def build(name, mols, pros, ys):
        p2, p3, p4 = fun(pros, 2, split_sequence), fun(pros, 3, split_sequence), fun(pros, 4, split_sequence)
        return TestbedDataset(root=str(proc_root), dataset=name, xd=np.asarray(mols), xt_2=np.asarray(p2),
                              xt_3=np.asarray(p3), xt_4=np.asarray(p4), y=np.asarray(ys), smile_graph=smile_graph)

    train_data = build(f"f{fold}_train", train_mol, train_pro, train_y)
    valid_data = build(f"f{fold}_valid", valid_mol, valid_pro, valid_y)
    test_data = build(f"f{fold}_test", test_mol, test_pro, test_y)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    sub = GraphEnhance(78, 128, 4, mode='TS', times=2)
    sub.load_state_dict(torch.load(args.sub_checkpoint, map_location='cpu'), strict=False)
    model = GINConvNet(sub).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid_mse = float('inf')
    best_state = None
    for epoch in range(args.epochs):
        train_one_epoch(model, device, train_loader, optimizer, loss_fn)
        g, p = predict(model, device, valid_loader)
        v_mse = mse(g, p)
        if v_mse < best_valid_mse:
            best_valid_mse = v_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    g_test, p_test = predict(model, device, test_loader)
    metrics = {
        "rmse": float(rmse(g_test, p_test)),
        "mse": float(mse(g_test, p_test)),
        "pearson": float(pearson(g_test, p_test)),
        "spearman": float(spearman(g_test, p_test)),
        "ci": float(ci(g_test, p_test)),
        "rm2": float(get_rm2(g_test, p_test)),
    }
    return metrics, g_test, p_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-raw-root", default="data_raw150")
    ap.add_argument("--proc-root", default="data_processed150")
    ap.add_argument("--sub-checkpoint", default="sub_50000.pth")
    ap.add_argument("--out-dir", default="results150")
    ap.add_argument("--pred-dir", default="predictions_submdta")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.0005)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.pred_dir).mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()
    for fold in range(args.folds):
        tf0 = time.time()
        metrics, g_test, p_test = run_fold(fold, args, device)
        elapsed = time.time() - tf0
        metrics["fold"] = fold
        metrics["seconds"] = elapsed
        results.append(metrics)
        print(f"fold {fold}: {metrics} ({elapsed:.1f}s)")

        np.savetxt(Path(args.pred_dir) / f"fold{fold}_pred.txt", p_test, fmt="%.6f")
        np.savetxt(Path(args.pred_dir) / f"fold{fold}_true.txt", g_test, fmt="%.6f")

        with open(Path(args.out_dir) / "cv_summary.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    rmses = [r["rmse"] for r in results]
    summary = {
        "per_fold": results,
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "total_seconds": time.time() - t0,
    }
    with open(Path(args.out_dir) / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== RESUMEN FINAL SubMDTA ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
