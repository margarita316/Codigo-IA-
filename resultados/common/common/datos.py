from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .descriptores import drug_descriptors, protein_descriptors

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cargar_dataset(nombre: str) -> pd.DataFrame:
    """nombre: 'DAVIS' o 'KIBA'. Devuelve columnas Drug_ID, Drug, Target_ID,
    Target, Y (afinidad ya en la escala continua estandar de la literatura:
    pKd para DAVIS, KIBA score para KIBA)."""
    nombre = nombre.upper()
    archivo = DATA_DIR / ("davis.tab" if nombre == "DAVIS" else "kiba.tab")
    if not archivo.exists():
        raise FileNotFoundError(
            f"No se encontro {archivo}. Ejecuta primero common/descargar_datos.py")
    df = pd.read_csv(archivo, sep="\t")
    df.columns = ["Drug_ID", "Drug", "Target_ID", "Target", "Y"]
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["Drug_ID", "Drug", "Target_ID", "Target", "Y"]).reset_index(drop=True)

    if nombre == "DAVIS":
        y = df["Y"].to_numpy(dtype=np.float64)
        if np.any(y <= 0):
            raise ValueError("DAVIS contiene Kd no positivos; no se puede convertir a pKd.")
        df["Y"] = 9.0 - np.log10(y)  # pKd, igual que en toda la literatura DTA

    # KIBA ya viene en la escala continua estandar (score KIBA de PyTDC), sin transformar.
    df["Drug_ID"] = df["Drug_ID"].astype(str)
    df["Target_ID"] = df["Target_ID"].astype(str)
    return df


def _hash_cache_key(nombre: str, kind: str) -> Path:
    return CACHE_DIR / f"{kind}_{nombre.upper()}.npz"


def descriptores_farmacos(df: pd.DataFrame, nombre_dataset: str) -> dict[str, np.ndarray]:
    """dict Drug_ID -> vector de descriptores, calculado una sola vez por
    farmaco unico y cacheado en disco (los pares repiten farmaco muchas veces)."""
    cache_path = _hash_cache_key(nombre_dataset, "drug")
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return dict(zip(data["ids"], data["vectors"]))

    unicos = df.drop_duplicates("Drug_ID")[["Drug_ID", "Drug"]]
    resultado = {}
    for drug_id, smiles in zip(unicos["Drug_ID"], unicos["Drug"]):
        resultado[drug_id] = drug_descriptors(smiles)

    ids = np.array(list(resultado.keys()), dtype=object)
    vectors = np.array(list(resultado.values()))
    np.savez(cache_path, ids=ids, vectors=vectors)
    return resultado


def descriptores_proteinas(df: pd.DataFrame, nombre_dataset: str) -> dict[str, np.ndarray]:
    cache_path = _hash_cache_key(nombre_dataset, "target")
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return dict(zip(data["ids"], data["vectors"]))

    unicos = df.drop_duplicates("Target_ID")[["Target_ID", "Target"]]
    resultado = {}
    for target_id, seq in zip(unicos["Target_ID"], unicos["Target"]):
        resultado[target_id] = protein_descriptors(seq)

    ids = np.array(list(resultado.keys()), dtype=object)
    vectors = np.array(list(resultado.values()))
    np.savez(cache_path, ids=ids, vectors=vectors)
    return resultado


def construir_matriz_pares(df: pd.DataFrame, drug_vecs: dict, target_vecs: dict):
    """Devuelve X_drug (n, d_drug), X_target (n, d_target), y (n,) alineados
    fila a fila con df. NO se normaliza aqui: la normalizacion se hace en
    cada fold, ajustada solo con las filas de entrenamiento (regla 9)."""
    x_drug = np.vstack([drug_vecs[d] for d in df["Drug_ID"]])
    x_target = np.vstack([target_vecs[t] for t in df["Target_ID"]])
    y = df["Y"].to_numpy(dtype=np.float64)
    return x_drug, x_target, y
