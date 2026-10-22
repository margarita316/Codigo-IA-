from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    repeat: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def cold_drug_splits(drug_ids: pd.Series, n_splits: int = 5, n_repeats: int = 10,
                      seed: int = 42):
    """Genera n_splits x n_repeats particiones agrupando por Drug_ID."""
    drug_ids = drug_ids.reset_index(drop=True)
    # np.asarray(..., dtype=object): rng.shuffle sobre un StringArray de
    # pandas (no es una Sequence real) puede duplicar elementos en vez de
    # permutarlos -- con un ndarray de objetos comun, shuffle es correcto.
    unique_drugs = np.asarray(drug_ids.unique(), dtype=object)
    rows_by_drug = {d: np.where(drug_ids.values == d)[0] for d in unique_drugs}

    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)
        shuffled = unique_drugs.copy()
        rng.shuffle(shuffled)
        chunks = np.array_split(shuffled, n_splits)
        for fold_i, test_drugs in enumerate(chunks):
            test_drug_set = set(test_drugs)
            test_idx = np.concatenate([rows_by_drug[d] for d in test_drugs])
            train_drugs = [d for d in unique_drugs if d not in test_drug_set]
            train_idx = np.concatenate([rows_by_drug[d] for d in train_drugs])
            yield Fold(repeat=repeat, fold=fold_i,
                       train_idx=np.sort(train_idx), test_idx=np.sort(test_idx))


def verificar_particion(df: pd.DataFrame, fold: Fold, drug_col: str = "Drug_ID") -> None:
    """Lanza AssertionError si algo del protocolo cold-drug se viola."""
    train_drugs = set(df.iloc[fold.train_idx][drug_col])
    test_drugs = set(df.iloc[fold.test_idx][drug_col])
    interseccion = train_drugs & test_drugs
    if interseccion:
        raise AssertionError(
            "Fuga cold-drug: " + str(len(interseccion)) + " farmacos comparten "
            "train/test en repeat=" + str(fold.repeat) + " fold=" + str(fold.fold))
    if set(fold.train_idx) & set(fold.test_idx):
        raise AssertionError("Indices de train y test se solapan.")


def verificar_duplicados(df: pd.DataFrame, drug_col: str = "Drug_ID",
                          drug_seq_col: str = "Drug", target_col: str = "Target_ID",
                          target_seq_col: str = "Target") -> dict:
    """Reporta duplicados exactos de pares, SMILES y secuencias (regla 10).
    No elimina nada por si mismo: informa para decision explicita."""
    reporte = {}
    pares_dup = df.duplicated(subset=[drug_col, target_col]).sum()
    reporte["pares_duplicados"] = int(pares_dup)

    smiles_por_id = df.drop_duplicates(drug_col)[[drug_col, drug_seq_col]]
    smiles_dup = smiles_por_id.duplicated(subset=[drug_seq_col]).sum()
    reporte["drug_id_con_smiles_duplicado"] = int(smiles_dup)

    seq_por_id = df.drop_duplicates(target_col)[[target_col, target_seq_col]]
    seq_dup = seq_por_id.duplicated(subset=[target_seq_col]).sum()
    reporte["target_id_con_secuencia_duplicada"] = int(seq_dup)

    return reporte
