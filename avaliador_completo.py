
import pandas as pd
from datetime import datetime
from utils.forecast_evaluation import (
    avaliar_previsao_real,
    avaliar_lstm_vs_real,
    avaliar_gatilho_atingido,
    avaliar_rsi_comportamento
)
import os

def executar_avaliacao_completa(ticker, intervalo, caminho_prev_prophet, caminho_prev_lstm, caminho_real, rsi_series, preco_series_rsi, gatilho=None, alvo=None, tipo=None):
    resultados = {}

    # Avaliação Prophet
    try:
        prev_prophet = pd.read_csv(caminho_prev_prophet)
        real = pd.read_csv(caminho_real)
        resultado_prophet = avaliar_previsao_real(prev_prophet, real, ticker, intervalo)
        resultados.update({f"MAPE_Prophet": resultado_prophet["MAPE"], "RMSE_Prophet": resultado_prophet["RMSE"]})
    except Exception as e:
        resultados.update({"MAPE_Prophet": None, "RMSE_Prophet": None})
        print(f"[Erro Prophet] {e}")

    # Avaliação LSTM
    try:
        df_lstm = pd.read_csv(caminho_prev_lstm)
        resultado_lstm = avaliar_lstm_vs_real(
            df_lstm['previsto_lstm'].tolist(),
            df_lstm['preco_real'].tolist(),
            df_lstm['ds'].tolist(),
            ticker, intervalo
        )
        resultados.update({f"MAPE_LSTM": resultado_lstm["MAPE"], "RMSE_LSTM": resultado_lstm["RMSE"]})
    except Exception as e:
        resultados.update({"MAPE_LSTM": None, "RMSE_LSTM": None})
        print(f"[Erro LSTM] {e}")

    # Avaliação de gatilho técnico
    if gatilho and alvo and tipo:
        try:
            preco_real_series = real['preco_real']
            atingido = avaliar_gatilho_atingido(preco_real_series, gatilho, alvo, tipo)
            resultados["Gatilho_atingido"] = atingido
        except Exception as e:
            resultados["Gatilho_atingido"] = None
            print(f"[Erro Gatilho] {e}")

    # Avaliação de RSI
    try:
        acertos_rsi = avaliar_rsi_comportamento(rsi_series, preco_series_rsi)
        resultados["RSI_acertos"] = acertos_rsi
    except Exception as e:
        resultados["RSI_acertos"] = None
        print(f"[Erro RSI] {e}")

    resultados["ticker"] = ticker
    resultados["intervalo"] = intervalo
    resultados["data"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_resultado = pd.DataFrame([resultados])
    caminho_saida = "avaliacoes/score_completo.csv"
    os.makedirs("avaliacoes", exist_ok=True)
    if not os.path.exists(caminho_saida):
        df_resultado.to_csv(caminho_saida, index=False)
    else:
        df_existente = pd.read_csv(caminho_saida)
        df_final = pd.concat([df_existente, df_resultado], ignore_index=True)
        df_final.to_csv(caminho_saida, index=False)

    return resultados

if __name__ == "__main__":
    # Exemplo de parâmetros (ajuste para o seu caso real!)
    import pandas as pd

    resultados = executar_avaliacao_completa(
        ticker="pendle-USD",
        intervalo="30min",
        caminho_prev_prophet="previsoes_prophet/prophet_pendleUSD.csv",
        caminho_prev_lstm="avaliacoes/lstm_pendle-USD.csv",
        caminho_real="avaliacoes/precos_reais_pendle-USD.csv",
        rsi_series=pd.Series([71, 73, 68, 65]),
        preco_series_rsi=pd.Series([4.50, 4.48, 4.45, 4.43]),
        gatilho=None,
        alvo=None,
        tipo=None
    )