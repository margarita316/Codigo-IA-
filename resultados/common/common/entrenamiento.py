from __future__ import annotations

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


class PreprocesadorFold:
    """Imputa NaN, elimina columnas de varianza cero y escala: TODO ajustado
    unicamente con las filas de entrenamiento del fold, luego aplicado tal
    cual a test (regla 9)."""

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.selector = VarianceThreshold(threshold=1e-8)
        self.scaler = StandardScaler()

    def fit_transform(self, x_train: np.ndarray) -> np.ndarray:
        x = self.imputer.fit_transform(x_train)
        x = self.selector.fit_transform(x)
        x = self.scaler.fit_transform(x)
        return x.astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self.imputer.transform(x)
        x = self.selector.transform(x)
        x = self.scaler.transform(x)
        return x.astype(np.float32)


def preparar_fold(x_drug, x_target, y, train_idx, test_idx):
    """Devuelve un dict con train/val/test, con imputacion/escalado
    ajustados solo en train, y separando ademas un 10% interno de
    VALIDACION (tambien sacado exclusivamente de train) para early
    stopping -- nunca se usa test para decidir cuando parar."""
    prep_d = PreprocesadorFold()
    prep_t = PreprocesadorFold()

    xd_train_raw, xd_test_raw = x_drug[train_idx], x_drug[test_idx]
    xt_train_raw, xt_test_raw = x_target[train_idx], x_target[test_idx]
    y_train_raw, y_test = y[train_idx], y[test_idx]

    xd_train_all = prep_d.fit_transform(xd_train_raw)
    xt_train_all = prep_t.fit_transform(xt_train_raw)
    xd_test = prep_d.transform(xd_test_raw)
    xt_test = prep_t.transform(xt_test_raw)

    rng = np.random.default_rng(0)
    n = len(train_idx)
    perm = rng.permutation(n)
    n_val = max(int(0.1 * n), 1)
    val_pos, tr_pos = perm[:n_val], perm[n_val:]

    return {
        "xd_tr": xd_train_all[tr_pos], "xt_tr": xt_train_all[tr_pos], "y_tr": y_train_raw[tr_pos],
        "xd_val": xd_train_all[val_pos], "xt_val": xt_train_all[val_pos], "y_val": y_train_raw[val_pos],
        "xd_test": xd_test, "xt_test": xt_test, "y_test": y_test,
    }


def entrenar_con_early_stopping(modelo, datos, lr, max_epochs, paciencia,
                                 batch_size, device="cpu", weight_decay=0.0,
                                 verbose=False):
    """Bucle de entrenamiento generico: Adam + MSE, con early stopping sobre
    el MSE del split de VALIDACION interno (nunca sobre test)."""
    modelo.to(device)
    optim = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    xd_tr = torch.tensor(datos["xd_tr"], dtype=torch.float32, device=device)
    xt_tr = torch.tensor(datos["xt_tr"], dtype=torch.float32, device=device)
    y_tr = torch.tensor(datos["y_tr"], dtype=torch.float32, device=device).view(-1, 1)
    xd_val = torch.tensor(datos["xd_val"], dtype=torch.float32, device=device)
    xt_val = torch.tensor(datos["xt_val"], dtype=torch.float32, device=device)
    y_val = torch.tensor(datos["y_val"], dtype=torch.float32, device=device).view(-1, 1)

    n = xd_tr.shape[0]
    mejor_val = float("inf")
    mejor_estado = None
    sin_mejora = 0

    for epoch in range(max_epochs):
        modelo.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optim.zero_grad()
            salida = modelo(xd_tr[idx], xt_tr[idx])
            extra = getattr(modelo, "ultima_perdida_extra", None)
            loss = loss_fn(salida, y_tr[idx])
            if extra is not None:
                loss = loss + extra
            loss.backward()
            optim.step()
            total_loss += float(loss.detach()) * len(idx)

        modelo.eval()
        with torch.no_grad():
            pred_val = modelo(xd_val, xt_val)
            val_loss = float(loss_fn(pred_val, y_val))

        if verbose:
            print("    epoch " + str(epoch).zfill(3) + " train_loss=" +
                  format(total_loss / n, ".4f") + " val_loss=" + format(val_loss, ".4f"))

        if val_loss < mejor_val - 1e-5:
            mejor_val = val_loss
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    return modelo


def predecir(modelo, x_drug, x_target, device="cpu", batch_size=2048):
    modelo.eval()
    salidas = []
    with torch.no_grad():
        for start in range(0, len(x_drug), batch_size):
            xd = torch.tensor(x_drug[start:start + batch_size], dtype=torch.float32, device=device)
            xt = torch.tensor(x_target[start:start + batch_size], dtype=torch.float32, device=device)
            salidas.append(modelo(xd, xt).cpu().numpy())
    return np.concatenate(salidas).reshape(-1)
