"""Split cold-drug v2: en vez de mezclar farmacos individuales al azar,
agrupa primero los farmacos QUIMICAMENTE PARECIDOS entre si (clustering de
Butina sobre fingerprints Morgan) y reparte CLUSTERS completos entre los 5
folds. Asi, dos farmacos casi identicos nunca quedan uno en train y otro en
test -- una prueba de generalizacion mas realista que agrupar por Drug_ID
al azar (que es lo que hacia v1, ver common/cv_frio.py).

El split se calcula UNA SOLA VEZ por dataset y se guarda en disco
(data/folds_<DATASET>_v2.json), para que los 3 metodos y las 4 variantes
de ablacion usen EXACTAMENTE el mismo split -- ninguna diferencia de
metricas entre corridas puede deberse a un split distinto.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from .cv_frio import Fold

RDLogger.DisableLog("rdApp.*")

FOLDS_DIR = Path(__file__).resolve().parent.parent / "data"


def _fingerprints(drug_id_to_smiles: dict) -> dict:
    fps = {}
    for drug_id, smiles in drug_id_to_smiles.items():
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            fps[drug_id] = None
            continue
        fps[drug_id] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return fps


def clusterizar_farmacos(drug_id_to_smiles: dict, corte_distancia: float = 0.4) -> dict:
    """Clustering de Butina por similitud de Tanimoto. corte_distancia=0.4
    significa que dos farmacos con Tanimoto >= 0.6 caen en el mismo cluster
    (umbral estandar en cheminformatica para 'estructuralmente similares').
    Devuelve {Drug_ID: cluster_id}. Los SMILES invalidos van cada uno a su
    propio cluster (no se puede agrupar lo que no se puede comparar)."""
    ids = list(drug_id_to_smiles.keys())
    fps = _fingerprints(drug_id_to_smiles)

    validos = [i for i in ids if fps[i] is not None]
    invalidos = [i for i in ids if fps[i] is None]

    asignacion = {}
    if validos:
        fps_validos = [fps[i] for i in validos]
        distancias = []
        for idx in range(1, len(fps_validos)):
            sims = DataStructs.BulkTanimotoSimilarity(fps_validos[idx], fps_validos[:idx])
            distancias.extend([1 - s for s in sims])
        clusters = Butina.ClusterData(distancias, len(fps_validos), corte_distancia, isDistData=True)
        for cluster_id, miembros in enumerate(clusters):
            for pos in miembros:
                asignacion[validos[pos]] = cluster_id
    siguiente_id = (max(asignacion.values()) + 1) if asignacion else 0
    for i in invalidos:
        asignacion[i] = siguiente_id
        siguiente_id += 1
    return asignacion


def _repartir_clusters_en_folds(cluster_to_drugs: dict, n_splits: int, rng: np.random.Generator):
    """Reparte clusters completos en n_splits grupos, balanceando la
    cantidad total de FARMACOS (no clusters) por grupo via un heuristico
    greedy: se recorren los clusters de mayor a menor tamano y cada uno se
    asigna al grupo mas chico hasta el momento (bin-packing clasico)."""
    cluster_ids = list(cluster_to_drugs.keys())
    orden = sorted(cluster_ids, key=lambda c: len(cluster_to_drugs[c]), reverse=True)
    # empate aleatorio entre clusters del mismo tamano, para variar entre repeticiones
    rng.shuffle(orden)
    orden = sorted(orden, key=lambda c: len(cluster_to_drugs[c]), reverse=True)

    grupos = [[] for _ in range(n_splits)]
    tamanos = [0] * n_splits
    for c in orden:
        destino = int(np.argmin(tamanos))
        grupos[destino].extend(cluster_to_drugs[c])
        tamanos[destino] += len(cluster_to_drugs[c])
    return grupos


def generar_folds_por_clustering(df: pd.DataFrame, n_splits: int = 5, n_repeats: int = 10,
                                  seed: int = 42, corte_distancia: float = 0.4):
    unicos = df.drop_duplicates("Drug_ID")[["Drug_ID", "Drug"]]
    drug_id_to_smiles = dict(zip(unicos["Drug_ID"], unicos["Drug"]))

    asignacion_cluster = clusterizar_farmacos(drug_id_to_smiles, corte_distancia)
    cluster_to_drugs: dict = {}
    for drug_id, cluster_id in asignacion_cluster.items():
        cluster_to_drugs.setdefault(cluster_id, []).append(drug_id)

    drug_ids = df["Drug_ID"].reset_index(drop=True)
    rows_by_drug = {d: np.where(drug_ids.values == d)[0] for d in drug_id_to_smiles}

    folds = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)
        grupos = _repartir_clusters_en_folds(cluster_to_drugs, n_splits, rng)
        for fold_i, test_drugs in enumerate(grupos):
            test_idx = np.concatenate([rows_by_drug[d] for d in test_drugs]) if test_drugs else np.array([], dtype=int)
            test_set = set(test_drugs)
            train_drugs = [d for d in drug_id_to_smiles if d not in test_set]
            train_idx = np.concatenate([rows_by_drug[d] for d in train_drugs])
            folds.append(Fold(repeat=repeat, fold=fold_i,
                               train_idx=np.sort(train_idx), test_idx=np.sort(test_idx)))
    metadata = {
        "n_farmacos": len(drug_id_to_smiles),
        "n_clusters": len(cluster_to_drugs),
        "corte_distancia": corte_distancia,
        "tamano_clusters": sorted([len(v) for v in cluster_to_drugs.values()], reverse=True),
    }
    return folds, metadata


def guardar_folds(folds: list, metadata: dict, nombre_dataset: str) -> Path:
    FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FOLDS_DIR / f"folds_{nombre_dataset.upper()}_v2.json"
    payload = {
        "metadata": metadata,
        "folds": [
            {"repeat": f.repeat, "fold": f.fold,
             "train_idx": f.train_idx.tolist(), "test_idx": f.test_idx.tolist()}
            for f in folds
        ],
    }
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return ruta


def cargar_folds(nombre_dataset: str):
    ruta = FOLDS_DIR / f"folds_{nombre_dataset.upper()}_v2.json"
    if not ruta.exists():
        return None, None
    with open(ruta, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    folds = [
        Fold(repeat=f["repeat"], fold=f["fold"],
             train_idx=np.array(f["train_idx"], dtype=int),
             test_idx=np.array(f["test_idx"], dtype=int))
        for f in payload["folds"]
    ]
    return folds, payload["metadata"]


def obtener_folds_v2(df: pd.DataFrame, nombre_dataset: str, n_splits: int = 5,
                      n_repeats: int = 10, seed: int = 42):
    """Carga el split guardado si existe; si no, lo genera y lo guarda.
    Garantiza que TODAS las corridas de v2 (los 3 metodos + las 4
    variantes de ablacion + el baseline) usan el mismo split exacto."""
    folds, metadata = cargar_folds(nombre_dataset)
    if folds is not None:
        return folds, metadata
    folds, metadata = generar_folds_por_clustering(df, n_splits, n_repeats, seed)
    guardar_folds(folds, metadata, nombre_dataset)
    return folds, metadata
