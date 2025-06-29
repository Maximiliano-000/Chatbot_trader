from lstm_forecaster import CriptoForecaster
import pandas as pd

TICKER = "pendle-USD"
DIAS = 5  # ou quantos você precisar

# Carregue o modelo LSTM treinado
forecaster = CriptoForecaster(TICKER)
forecaster.carregar_modelo_treinado()

# Gere as previsões LSTM
previsoes = forecaster.prever(dias=DIAS)
previstos = [p["valor"] for p in previsoes]

# Carregue os preços reais (últimos DIAS)
real = pd.read_csv("avaliacoes/precos_reais_pendle-USD.csv")
real_alinhado = real.tail(DIAS).reset_index(drop=True)

# Monte o DataFrame final
df_lstm = pd.DataFrame({
    "ds": real_alinhado["ds"],
    "previsto_lstm": previstos,
    "preco_real": real_alinhado["preco_real"]
})

df_lstm.to_csv("avaliacoes/lstm_pendle-USD.csv", index=False)
print("✅ Arquivo salvo: avaliacoes/lstm_pendle-USD.csv")
