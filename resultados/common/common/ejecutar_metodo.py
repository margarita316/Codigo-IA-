from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cv_frio import cold_drug_splits, verificar_duplicados, verificar_particion
from .datos import cargar_dataset, construir_matriz_pares, descriptores_farmacos, descriptores_proteinas
from .entrenamiento import entrenar_con_early_stopping, predecir, preparar_fold
from .metricas import calcular_metricas

RESULT_DIR = Path(__file__).resolve().parent.parent / "resultados"


def ejecutar_metodo(nombre_metodo: str, crear_modelo, hparams: dict,
                     datasets=("DAVIS", "KIBA"), n_splits: int = 5, n_repeats: int = 10,
                     seed: int = 42, verbose_epochs: bool = False):
    metodo_dir = RESULT_DIR / nombre_metodo
    metodo_dir.mkdir(parents=True, exist_ok=True)

    for nombre_ds in datasets:
        print("=" * 70)
        print(f"{nombre_metodo} | dataset={nombre_ds}")
        print("=" * 70)

        df = cargar_dataset(nombre_ds)
        dup = verificar_duplicados(df)
        print("Verificacion de duplicados:", dup)

        drug_vecs = descriptores_farmacos(df, nombre_ds)
        target_vecs = descriptores_proteinas(df, nombre_ds)
        x_drug, x_target, y = construir_matriz_pares(df, drug_vecs, target_vecs)
        print(f"Pares: {len(df)} | dim_farmaco={x_drug.shape[1]} | dim_proteina={x_target.shape[1]}")

        filas_resultado = []
        t0 = time.time()
        for fold in cold_drug_splits(df["Drug_ID"], n_splits=n_splits, n_repeats=n_repeats, seed=seed):
            verificar_particion(df, fold)  # regla 10: aborta si hay fuga cold-drug

            datos_fold = preparar_fold(x_drug, x_target, y, fold.train_idx, fold.test_idx)
            modelo = crear_modelo(datos_fold["xd_tr"].shape[1], datos_fold["xt_tr"].shape[1])
            modelo = entrenar_con_early_stopping(
                modelo, datos_fold,
                lr=hparams["lr"], max_epochs=hparams["max_epochs"],
                paciencia=hparams["paciencia"], batch_size=hparams["batch_size"],
                weight_decay=hparams.get("weight_decay", 0.0), verbose=verbose_epochs,
            )
            y_pred = predecir(modelo, datos_fold["xd_test"], datos_fold["xt_test"])
            m = calcular_metricas(datos_fold["y_test"], y_pred)
            m.update({"repeat": fold.repeat, "fold": fold.fold, "n_test": len(fold.test_idx)})
            filas_resultado.append(m)

            transcurrido = time.time() - t0
            print(f"  repeat={fold.repeat:02d} fold={fold.fold} "
                  f"CI={m['CI']:.4f} RMSE={m['RMSE']:.4f} MSE={m['MSE']:.4f} R2={m['R2']:.4f} "
                  f"(t={transcurrido:.0f}s)")

        tabla = pd.DataFrame(filas_resultado)
        tabla.to_csv(metodo_dir / f"metricas_por_fold_{nombre_ds}.csv", index=False)
        resumen = tabla[["CI", "RMSE", "MSE", "R2"]].agg(["mean", "std"]).T
        resumen.to_csv(metodo_dir / f"resumen_{nombre_ds}.csv")

        print(f"\n=== {nombre_metodo} | {nombre_ds} | media +/- desviacion estandar ({n_splits}x{n_repeats} folds) ===")
        for metrica, fila in resumen.iterrows():
            print(f"{metrica:6s}: {fila['mean']:.4f} +/- {fila['std']:.4f}")

        with open(metodo_dir / f"config_{nombre_ds}.json", "w", encoding="utf-8") as f:
            json.dump({
                "metodo": nombre_metodo, "dataset": nombre_ds,
                "n_pares": len(df), "n_splits": n_splits, "n_repeats": n_repeats,
                "seed": seed, "hiperparametros": hparams,
                "duplicados": dup,
            }, f, ensure_ascii=False, indent=2)
