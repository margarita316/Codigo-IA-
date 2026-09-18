from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .descriptores import BLOQUES_FARMACO_V2, BLOQUES_PROTEINA_V2, rangos_de_bloques
from .preentrenamiento import preentrenar_y_aumentar

CAP_POR_BLOQUE = 64


@dataclass
class ConfigAblacion:
    nombre: str
    bloques_farmaco: list
    bloques_proteina: list
    usar_preentrenamiento: bool = False


BASE = ConfigAblacion("base", ["rdkit_descriptores"], ["aac", "dpc", "ctd", "globales"])
MAS_MORGAN = ConfigAblacion("mas_morgan", ["rdkit_descriptores", "morgan_fingerprint"],
                             ["aac", "dpc", "ctd", "globales"])
MAS_POSICIONAL = ConfigAblacion("mas_posicional", ["rdkit_descriptores", "morgan_fingerprint"],
                                 ["aac", "dpc", "ctd", "globales", "posicional_tercios"])
COMPLETA = ConfigAblacion("completa", ["rdkit_descriptores", "morgan_fingerprint"],
                           ["aac", "dpc", "ctd", "globales", "posicional_tercios"],
                           usar_preentrenamiento=True)


def recortar_por_bloques(x_full, bloques_meta, incluir):
    rangos = rangos_de_bloques(bloques_meta)
    columnas = []
    for nombre in incluir:
        ini, fin = rangos[nombre]
        columnas.append(x_full[:, ini:fin])
    return np.concatenate(columnas, axis=1)


class PreprocesadorBloques:
    """Igual que PreprocesadorFold (imputar+varianza+escalar) pero con un
    SelectKBest AJUSTADO POR SEPARADO dentro de cada bloque, para no
    mezclar columnas de bloques distintos en la seleccion y mantener el
    significado de cada bloque para los metodos que lo necesitan."""

    def __init__(self, bloques_incluidos):
        self.bloques_incluidos = bloques_incluidos
        self.imputer = SimpleImputer(strategy="median")
        self.selector_varianza = VarianceThreshold(threshold=1e-8)
        self.selectores_k = {}
        self.scaler = StandardScaler()
        self.bloques_finales = []

    def fit_transform(self, x_train, y_train):
        x = self.imputer.fit_transform(x_train)
        x = self.selector_varianza.fit_transform(x)

        mascara = self.selector_varianza.get_support()
        limites_originales = []
        inicio = 0
        for nombre, tamano in self.bloques_incluidos:
            limites_originales.append((nombre, inicio, inicio + tamano))
            inicio += tamano
        tamanos_tras_varianza = []
        for nombre, ini, fin in limites_originales:
            tamanos_tras_varianza.append((nombre, int(mascara[ini:fin].sum())))

        partes = []
        inicio = 0
        self.bloques_finales = []
        for nombre, tamano in tamanos_tras_varianza:
            bloque_x = x[:, inicio:inicio + tamano]
            inicio += tamano
            if tamano == 0:
                continue
            k = min(tamano, CAP_POR_BLOQUE)
            if k < tamano:
                selector = SelectKBest(mutual_info_regression, k=k)
                bloque_x = selector.fit_transform(bloque_x, y_train)
                self.selectores_k[nombre] = selector
            else:
                self.selectores_k[nombre] = None
            partes.append(bloque_x)
            self.bloques_finales.append((nombre, bloque_x.shape[1]))

        x_final = np.concatenate(partes, axis=1) if partes else x
        x_final = self.scaler.fit_transform(x_final)
        return x_final.astype(np.float32)

    def transform(self, x):
        x = self.imputer.transform(x)
        x = self.selector_varianza.transform(x)

        limites_originales = []
        inicio = 0
        for nombre, tamano in self.bloques_incluidos:
            limites_originales.append((nombre, inicio, inicio + tamano))
            inicio += tamano
        mascara = self.selector_varianza.get_support()
        tamanos_tras_varianza = []
        for nombre, ini, fin in limites_originales:
            tamanos_tras_varianza.append((nombre, int(mascara[ini:fin].sum())))

        partes = []
        inicio = 0
        for nombre, tamano in tamanos_tras_varianza:
            bloque_x = x[:, inicio:inicio + tamano]
            inicio += tamano
            if tamano == 0:
                continue
            selector = self.selectores_k.get(nombre)
            if selector is not None:
                bloque_x = selector.transform(bloque_x)
            partes.append(bloque_x)

        x_final = np.concatenate(partes, axis=1) if partes else x
        return self.scaler.transform(x_final).astype(np.float32)


def preparar_fold_v2(x_drug_full, x_target_full, y, train_idx, test_idx, config):
    """Version v2 de preparar_fold: recorta por bloques segun la
    configuracion de ablacion, preprocesa (imputar+varianza+SelectKBest
    por bloque+escalar) ajustado solo en train, opcionalmente preentrena
    autoencoders y aumenta con el embedding, y separa el 10% de validacion
    interna -- todo dentro de train_idx, test_idx nunca interviene."""
    xd_drug = recortar_por_bloques(x_drug_full, BLOQUES_FARMACO_V2, config.bloques_farmaco)
    xd_target = recortar_por_bloques(x_target_full, BLOQUES_PROTEINA_V2, config.bloques_proteina)

    bloques_farmaco_meta = [(n, s) for n, s in BLOQUES_FARMACO_V2 if n in config.bloques_farmaco]
    bloques_proteina_meta = [(n, s) for n, s in BLOQUES_PROTEINA_V2 if n in config.bloques_proteina]

    xd_train_raw, xd_test_raw = xd_drug[train_idx], xd_drug[test_idx]
    xt_train_raw, xt_test_raw = xd_target[train_idx], xd_target[test_idx]
    y_train_raw, y_test = y[train_idx], y[test_idx]

    prep_d = PreprocesadorBloques(bloques_farmaco_meta)
    prep_t = PreprocesadorBloques(bloques_proteina_meta)
    xd_train_all = prep_d.fit_transform(xd_train_raw, y_train_raw)
    xt_train_all = prep_t.fit_transform(xt_train_raw, y_train_raw)
    xd_test = prep_d.transform(xd_test_raw)
    xt_test = prep_t.transform(xt_test_raw)

    if config.usar_preentrenamiento:
        xd_unico_train = np.unique(xd_train_all, axis=0)
        xt_unico_train = np.unique(xt_train_all, axis=0)
        xd_train_all, xd_test = preentrenar_y_aumentar(xd_unico_train, xd_train_all, xd_test)
        xt_train_all, xt_test = preentrenar_y_aumentar(xt_unico_train, xt_train_all, xt_test)

    rng = np.random.default_rng(0)
    n = len(train_idx)
    perm = rng.permutation(n)
    n_val = max(int(0.1 * n), 1)
    val_pos, tr_pos = perm[:n_val], perm[n_val:]

    return {
        "xd_tr": xd_train_all[tr_pos], "xt_tr": xt_train_all[tr_pos], "y_tr": y_train_raw[tr_pos],
        "xd_val": xd_train_all[val_pos], "xt_val": xt_train_all[val_pos], "y_val": y_train_raw[val_pos],
        "xd_test": xd_test, "xt_test": xt_test, "y_test": y_test,
        "bloques_farmaco": prep_d.bloques_finales, "bloques_proteina": prep_t.bloques_finales,
    }
