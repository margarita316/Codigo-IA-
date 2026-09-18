from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DIM_EMBEDDING = 128


class _Autoencoder(nn.Module):
    def __init__(self, dim_entrada, dim_oculta=256, dim_embedding=DIM_EMBEDDING):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_entrada, dim_oculta), nn.ReLU(),
            nn.Linear(dim_oculta, dim_embedding),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_embedding, dim_oculta), nn.ReLU(),
            nn.Linear(dim_oculta, dim_entrada),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def entrenar_autoencoder(x_unico, epochs=40, lr=1e-3, batch_size=64, device="cpu"):
    """Entrena un autoencoder de reconstruccion sobre x_unico (una fila por
    entidad UNICA -- farmaco o proteina -- del train del fold, ya
    preprocesada). No usa Y en ningun momento."""
    modelo = _Autoencoder(x_unico.shape[1]).to(device)
    optim = torch.optim.Adam(modelo.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    x = torch.tensor(x_unico, dtype=torch.float32, device=device)
    n = x.shape[0]
    batch_size = min(batch_size, max(n, 1))

    modelo.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optim.zero_grad()
            recon, _ = modelo(x[idx])
            loss = loss_fn(recon, x[idx])
            loss.backward()
            optim.step()
    modelo.eval()
    return modelo


def calcular_embeddings(modelo, x, device="cpu", batch_size=2048):
    """Aplica el encoder ya entrenado a CUALQUIER vector de descriptores
    (train o test) -- igual que aplicar un StandardScaler ya ajustado."""
    modelo.eval()
    salidas = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            _, z = modelo(xb)
            salidas.append(z.cpu().numpy())
    return np.concatenate(salidas, axis=0)


def preentrenar_y_aumentar(x_train_unico, x_train_filas, x_test_filas, epochs=40):
    """Flujo completo: entrena el autoencoder solo con las entidades unicas
    de train, y devuelve (train, test) con el embedding concatenado a cada
    fila. x_train_unico: una fila por entidad UNICA de train (para entrenar
    el autoencoder). x_train_filas/x_test_filas: una fila POR PAR (para
    generar el embedding que efectivamente se usa como feature)."""
    modelo = entrenar_autoencoder(x_train_unico, epochs=epochs)
    emb_train = calcular_embeddings(modelo, x_train_filas)
    emb_test = calcular_embeddings(modelo, x_test_filas)
    aumentado_train = np.concatenate([x_train_filas, emb_train], axis=1)
    aumentado_test = np.concatenate([x_test_filas, emb_test], axis=1)
    return aumentado_train.astype(np.float32), aumentado_test.astype(np.float32)
