import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def baixar_e_validar(ticker, periodo="6mo", min_candles=100):
    dados = yf.download(ticker, period=periodo)
    if dados.empty or len(dados) < min_candles:
        print(f"⚠️ Dados insuficientes para {ticker}!")
        return None

    print(f"\n✅ Últimos 5 registros para {ticker}:")
    print(dados.tail())

    if dados.isnull().sum().any():
        print(f"⚠️ Atenção: Há valores NaN nos dados de {ticker}!")

    # Padronizando
    scaler = MinMaxScaler()
    dados_scaled = scaler.fit_transform(dados[['Close']].dropna())
    print(f"\n📈 Dados padronizados para {ticker} (últimos 5):")
    print(np.round(dados_scaled[-5:], 4))

    return dados_scaled

# Executando para SOL-USD e PENDLE-USD
sol_scaled = baixar_e_validar("SOL-USD")
pendle_scaled = baixar_e_validar("PENDLE-USD")