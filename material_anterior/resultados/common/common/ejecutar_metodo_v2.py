from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cv_frio import verificar_duplicados, verificar_particion
from .datos import cargar_dataset
from .datos_v2 import construir_matriz_pares_v2, descriptores_farmacos_v2, descriptores_proteinas_v2
from .entrenamiento import entrenar_con_early_stopping, predecir
from .entrenamiento_v2 import preparar_fold_v2
from .metricas import calcular_metricas
from .split_frio import obtener_folds_v2

RESULT_DIR = Path(__file__).resolve().parent.parent / "resultados"


def _versiones_librerias():
    versiones = {"python": platform.python_version()}
    for lib in ("torch", "numpy", "pandas", "sklearn", "rdkit"):
        try:
            mod = __import__(lib)
            versiones[lib] = getattr(mod, "__version__", "desconocida")
        except Exception:
            versiones[lib] = "no instalada"
    return versiones


def ejecutar_v2(nombre_corrida, crear_modelo, hparams, config_ablacion,
                 dataset="DAVIS", n_splits=5, n_repeats=10, seed=42, verbose_epochs=False):
    carpeta = RESULT_DIR / "v2" / nombre_corrida
    carpeta.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"{nombre_corrida} | dataset={dataset} | ablacion={config_ablacion.nombre}")
    print("=" * 70)

    df = cargar_dataset(dataset)
    dup = verificar_duplicados(df)
    print("Verificacion de duplicados:", dup)

    drug_vecs = descriptores_farmacos_v2(df, dataset)
    target_vecs = descriptores_proteinas_v2(df, dataset)
    x_drug_full, x_target_full, y = construir_matriz_pares_v2(df, drug_vecs, target_vecs)
    print(f"Pares: {len(df)} | dim_farmaco_completo={x_drug_full.shape[1]} "
          f"| dim_proteina_completo={x_target_full.shape[1]}")

    folds, meta_split = obtener_folds_v2(df, dataset, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    print("Split fijo (clustering quimico):", meta_split)

    filas_resultado = []
    t0 = time.time()
    for fold in folds:
        verificar_particion(df, fold)

        datos_fold = preparar_fold_v2(x_drug_full, x_target_full, y,
                                       fold.train_idx, fold.test_idx, config_ablacion)
        modelo = crear_modelo(datos_fold["xd_tr"].shape[1], datos_fold["xt_tr"].shape[1],
                               datos_fold["bloques_farmaco"], datos_fold["bloques_proteina"])
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
    tabla.to_csv(carpeta / f"metricas_por_fold_{dataset}.csv", index=False)
    resumen = tabla[["CI", "RMSE", "MSE", "R2"]].agg(["mean", "std"]).T
    resumen.to_csv(carpeta / f"resumen_{dataset}.csv")

    print(f"\n=== {nombre_corrida} | {dataset} | media +/- desviacion estandar "
          f"({n_splits}x{n_repeats} folds) ===")
    for metrica, fila in resumen.iterrows():
        print(f"{metrica:6s}: {fila['mean']:.4f} +/- {fila['std']:.4f}")

    with open(carpeta / f"protocolo_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump({
            "nombre_corrida": nombre_corrida,
            "dataset": dataset,
            "n_pares": len(df),
            "n_splits": n_splits, "n_repeats": n_repeats, "seed": seed,
            "config_ablacion": {
                "nombre": config_ablacion.nombre,
                "bloques_farmaco": config_ablacion.bloques_farmaco,
                "bloques_proteina": config_ablacion.bloques_proteina,
                "usar_preentrenamiento": config_ablacion.usar_preentrenamiento,
            },
            "hiperparametros": hparams,
            "duplicados": dup,
            "split_metadata": meta_split,
            "versiones_librerias": _versiones_librerias(),
        }, f, ensure_ascii=False, indent=2)

    return tabla, resumen
