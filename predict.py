import numpy as np
import os
from tensorflow.keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from db import salvar_previsao
import pandas as pd

def calcular_sma(dados, janela=20):
    """Calcula a média móvel simples (SMA)."""
    return pd.Series(dados.flatten()).rolling(window=janela).mean().iloc[-1]

def prever_proximo_fechamento(ticker, janela=60, period='1y', ajustar_com_sma=True, peso_sma=0.3):
    """
    Previsão do próximo preço com LSTM, ajustado por SMA20 com peso opcional.
    """
    try:
        is_cripto = "-USD" in ticker
        modelo_path = f"models/modelo_lstm{'_cripto' if is_cripto else ''}.h5"

        if not os.path.exists(modelo_path):
            print(f"❌ Modelo não encontrado: {modelo_path}")
            return None

        modelo = load_model(modelo_path)

        dados_brutos = yf.download(ticker, period=period, progress=False)['Close'].dropna().values.reshape(-1, 1)

        if len(dados_brutos) < janela:
            print(f"⚠️ Dados insuficientes para {ticker} ({len(dados_brutos)} pontos).")
            return None

        scaler = MinMaxScaler()
        dados_norm = scaler.fit_transform(dados_brutos)
        dados_recentes = np.reshape(dados_norm[-janela:], (1, janela, 1))

        previsao_norm = modelo.predict(dados_recentes)
        previsao_norm = np.clip(previsao_norm, a_min=0, a_max=None)

        previsao = scaler.inverse_transform(previsao_norm)[0][0]

        # Prints adicionais para debug visual
        print(f"[DEBUG] Valor normalizado previsto: {previsao_norm[0][0]}")
        print(f"[DEBUG] Valor invertido (original): {previsao:.2f}")

        if ajustar_com_sma:
            sma20 = calcular_sma(dados_brutos, janela=20)
            if not np.isnan(sma20):
                previsao_ajustada = previsao * (1 - peso_sma) + sma20 * peso_sma
                print(f"[DEBUG] SMA20: {sma20:.2f}")
                print(f"🔄 Previsão ajustada pela SMA20: {previsao:.2f} → {previsao_ajustada:.2f}")
                previsao = previsao_ajustada
            else:
                print("⚠️ SMA20 resultou em NaN, ajuste não realizado.")

        return previsao

    except Exception as e:
        print(f"🚨 Erro ao prever {ticker}: {str(e)}")
        return None

# Teste completo com salvar_previsao
if __name__ == "__main__":
    tickers = [
        "MRFG3", "SUZB3", "EGIE3", "CMIG4", "JBSS3", "BRFS3",
        "WEGE3", "CSAN3", "POMO4", "COGN3", "CYRE3", "CSMG3",
        "SOL-USD", "PENDLE-USD"
    ]

    for ticker in tickers:
        ticker_yf = ticker if "-USD" in ticker else f"{ticker}.SA"
        nome_amigavel = ticker.replace("-USD", "").replace(".SA", "")
        valor = prever_proximo_fechamento(ticker_yf, janela=20, period="6mo", ajustar_com_sma=True)

        if valor:
            moeda = "US$" if "-USD" in ticker else "R$"
            print(f"📈 Previsão final para {nome_amigavel}: {moeda} {valor:.2f}")

            # Salvar previsão no banco de dados
            salvar_previsao(
                user_id="simulado",
                ticker=nome_amigavel,
                period="6mo",
                janela=20,
                previsao=round(valor, 2)
            )
