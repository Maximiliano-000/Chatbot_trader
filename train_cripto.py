import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from model import criar_modelo
from utils.dados_com_fallback import obter_dados_com_fallback

# Parâmetros
tickers = ["PENDLE-USD"]
janela = 60
epochs = 100
output_dir = "modelos_lstm"
os.makedirs(output_dir, exist_ok=True)

for ticker in tickers:
    print(f"\n🚀 Treinando modelo para {ticker}...")

    # Coleta de dados via fallback (prioriza yfinance para treino)
    df, fonte, intervalo_usado, msg = obter_dados_com_fallback(
        ticker=ticker,
        intervalo="1d",
        periodo="1y",
        preferencia="yahoo"
    )

    if df.empty or "Close" not in df.columns or df["Close"].dropna().shape[0] < janela + 1:
        print(f"⚠️ Dados insuficientes para {ticker}. Pulando.")
        continue

    closes = df["Close"].dropna().values.reshape(-1, 1)

    # Normalização
    scaler = MinMaxScaler()
    dados_norm = scaler.fit_transform(closes)

    # Preparação das janelas de treino
    X, y = [], []
    for i in range(janela, len(dados_norm)):
        X.append(dados_norm[i-janela:i, 0])
        y.append(dados_norm[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], janela, 1))

    # Modelo e treinamento
    modelo = criar_modelo(janela, 1)
    modelo.fit(X, y, epochs=epochs, batch_size=32, verbose=1)

    # Salvamento
    nome_base = ticker.lower().replace("-", "_").replace("/", "_")
    modelo_path = f"{output_dir}/{nome_base}_modelo.h5"
    scaler_path = f"{output_dir}/{nome_base}_scaler.pkl"

    modelo.save(modelo_path)
    pd.to_pickle(scaler, scaler_path)

    print(f"✅ Modelo salvo em: {modelo_path}")
    print(f"✅ Scaler salvo em: {scaler_path}")

