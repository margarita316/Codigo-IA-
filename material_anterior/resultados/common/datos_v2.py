"""Carga y cacheo de los descriptores v2 (con Morgan fingerprint y
composicion por tercios incluidos). Igual que common/datos.py pero usando
drug_descriptors_v2 / protein_descriptors_v2."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .descriptores import drug_descriptors_v2, protein_descriptors_v2

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(nombre_dataset: str, kind: str) -> Path:
    return CACHE_DIR / f"{kind}_{nombre_dataset.upper()}_v2.npz"


def descriptores_farmacos_v2(df: pd.DataFrame, nombre_dataset: str) -> dict:
    cache_path = _cache_path(nombre_dataset, "drug")
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return dict(zip(data["ids"], data["vectors"]))

    unicos = df.drop_duplicates("Drug_ID")[["Drug_ID", "Drug"]]
    resultado = {}
    for drug_id, smiles in zip(unicos["Drug_ID"], unicos["Drug"]):
        resultado[drug_id] = drug_descriptors_v2(smiles)

    ids = np.array(list(resultado.keys()), dtype=object)
    vectors = np.array(list(resultado.values()))
    np.savez(cache_path, ids=ids, vectors=vectors)
    return resultado


def descriptores_proteinas_v2(df: pd.DataFrame, nombre_dataset: str) -> dict:
    cache_path = _cache_path(nombre_dataset, "target")
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return dict(zip(data["ids"], data["vectors"]))

    unicos = df.drop_duplicates("Target_ID")[["Target_ID", "Target"]]
    resultado = {}
    for target_id, seq in zip(unicos["Target_ID"], unicos["Target"]):
        resultado[target_id] = protein_descriptors_v2(seq)

    ids = np.array(list(resultado.keys()), dtype=object)
    vectors = np.array(list(resultado.values()))
    np.savez(cache_path, ids=ids, vectors=vectors)
    return resultado


def construir_matriz_pares_v2(df: pd.DataFrame, drug_vecs: dict, target_vecs: dict):
    x_drug = np.vstack([drug_vecs[d] for d in df["Drug_ID"]])
    x_target = np.vstack([target_vecs[t] for t in df["Target_ID"]])
    y = df["Y"].to_numpy(dtype=np.float64)
    return x_drug, x_target, y
