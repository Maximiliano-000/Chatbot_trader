# pyright: reportMissingImports=false
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from lstm_forecaster import CriptoForecaster
from logger_perda import LoggerDePerda
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Ativos para testar
tickers = ["PENDLE-USD", "BTC-USD"]

# Parâmetros
janela = 60
dias = 5
epochs = 100

# Diretórios
os.makedirs("resultados_lstm", exist_ok=True)
metrics_path = "resultados_lstm/lstm_metrics.csv"
if not os.path.exists(metrics_path):
    with open(metrics_path, "w") as f:
        f.write("Ticker,RMSE,MAE,MAPE\n")

# Loop por ticker
for ticker in tickers:
    try:
        print(f"\n🚀 Treinando LSTM para {ticker}...")

        forecaster = CriptoForecaster(ticker, janela=janela, epochs=epochs)
        forecaster.carregar_dados()  # usa preferencia="yahoo" internamente

        early = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        forecaster.treinar(callbacks=[early, LoggerDePerda()])

        previsoes = forecaster.prever(dias=dias)
        valores_previstos = np.array([p["valor"] for p in previsoes])

        serie_real = forecaster.scaler.inverse_transform(
            forecaster.dados_treinamento[-(dias + janela):].reshape(-1, 1)
        ).flatten()[-dias:]

        rmse = mean_squared_error(serie_real, valores_previstos, squared=False)
        mae = mean_absolute_error(serie_real, valores_previstos)
        mape = mean_absolute_percentage_error(serie_real, valores_previstos)

        print(f"✅ {ticker}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.4f}")

        df = pd.DataFrame({
            "Dia": [f"D{i+1}" for i in range(dias)],
            "Real": serie_real,
            "Previsto": valores_previstos
        })
        nome_curto = ticker.replace("-", "").replace(".", "")
        df.to_csv(f"resultados_lstm/lstm_{nome_curto}.csv", index=False)

        with open(metrics_path, "a") as f:
            f.write(f"{ticker},{rmse:.2f},{mae:.2f},{mape:.4f}\n")

        plt.figure(figsize=(10, 5))
        plt.plot(df["Dia"], df["Real"], label="Real", marker="o")
        plt.plot(df["Dia"], df["Previsto"], label="Previsto", marker="x")
        plt.title(f"📈 Previsão LSTM – {ticker}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(f"resultados_lstm/lstm_{nome_curto}.png")
        plt.close()

    except Exception as e:
        print(f"❌ Falha com {ticker}: {e}")

    time.sleep(10)