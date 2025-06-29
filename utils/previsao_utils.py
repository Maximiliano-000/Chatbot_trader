import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def preencher_volume_futuro(df_prophet, futuro, metodo="auto", dias=None):
    """
    Preenche a coluna 'Volume' no DataFrame 'futuro' com valores previstos com base
    no histórico de 'Volume' em df_prophet. Usa regressão linear simples ou fallback.
    """
    # ✅ Segurança: se Volume não existir ou tiver poucos dados
    if "Volume" not in df_prophet.columns or df_prophet["Volume"].dropna().shape[0] < 5:
        volume_constante = 0.0
        futuro["Volume"] = volume_constante
        return futuro

    volume = df_prophet["Volume"].dropna().tail(30)

    if metodo == "media":
        volume_previsto = [volume.mean()] * len(futuro)
    elif metodo == "ultimo":
        volume_previsto = [volume.iloc[-1]] * len(futuro)
    else:  # "auto" ou "regressao"
        X = np.arange(len(volume)).reshape(-1, 1)
        y = volume.values.reshape(-1, 1)

        if np.std(y) < 1e-3:
            volume_previsto = [volume.iloc[-1]] * len(futuro)
        else:
            modelo = LinearRegression().fit(X, y)
            dias_futuros = len(futuro) - len(df_prophet) if dias is None else dias
            X_futuro = np.arange(len(volume), len(volume) + dias_futuros).reshape(-1, 1)
            volume_previsto = modelo.predict(X_futuro).flatten()

    futuro["Volume"] = volume_previsto
    return futuro