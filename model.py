# model.py

import re
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analise_com_gpt(ticker, df, previsao_df):
    ultimos_precos = df['Close'].tail(3).values.tolist()
    rsi_atual = df['RSI'].iloc[-1]
    sma20 = df['SMA20'].iloc[-1]
    upper_band = df['UpperBand'].iloc[-1]
    lower_band = df['LowerBand'].iloc[-1]

    valores_previstos = previsao_df['yhat'].tail(5).tolist()
    valores_filtrados = [max(0, v) for v in valores_previstos]
    valores_formatados = ', '.join([f"R$ {v:.2f}" for v in valores_filtrados])

    prompt = f"""
Você é um analista técnico. Gere uma análise clara, com os seguintes blocos:

📊 Indicadores Técnicos  
📉 Tendência Atual  
🔮 Cenário Projetado  
📌 Estratégia  
🛡 Gestão de Risco  

Inclua este aviso no final:  
> _Esta análise é gerada automaticamente com base em indicadores públicos. Não constitui recomendação de investimento._

Dados do ativo {ticker}:
- Últimos preços: {ultimos_precos}
- RSI atual: {rsi_atual:.2f}
- SMA20: {sma20:.2f}
- Bollinger: sup = {upper_band:.2f}, inf = {lower_band:.2f}
- Previsão: {valores_formatados}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    texto = response.choices[0].message.content

    partes = re.split(r"(?:📊|📉|🔮|📌|🛡)", texto)
    partes = [p.strip() for p in partes if p.strip()]

    return {
        "indicadores": partes[0] if len(partes) > 0 else "",
        "tendencia": partes[1] if len(partes) > 1 else "",
        "previsao": partes[2] if len(partes) > 2 else "",
        "estrategia": partes[3] if len(partes) > 3 else "",
        "risco": partes[4] if len(partes) > 4 else "",
        "aviso": "_Esta análise é gerada automaticamente com base em indicadores técnicos públicos. Não constitui recomendação de investimento._"
    }

def analise_fallback():
    return {
        "indicadores": "_Indicadores indisponíveis._",
        "tendencia": "Tendência não determinada.",
        "previsao": "Sem projeção disponível.",
        "estrategia": "Sem sugestão de estratégia.",
        "risco": "Gestão de risco não avaliada.",
        "aviso": "_Esta análise está incompleta devido a erro interno._"
    }

import numpy as np
from flask import session

# =============================================================================
# Ajuste técnico para previsão LSTM
# =============================================================================

def ajustar_previsao_lstm(valor_original, indicadores, ticker=None):
    try:
        # Validação explícita do SMA20
        if 'SMA20' not in indicadores or indicadores['SMA20'].isna().iloc[-1]:
            print("[⚠️ Aviso] SMA20 não encontrado ou inválido.")
            return valor_original

        sma20 = indicadores['SMA20'].iloc[-1]

        # Mistura leve e segura da previsão original com SMA20
        valor_ajustado = (valor_original * 0.8 + sma20 * 0.2)

        # Aplica limites técnicos (±15% SMA20)
        limite_superior = sma20 * 1.15
        limite_inferior = sma20 * 0.85
        valor_ajustado = float(np.clip(valor_ajustado, limite_inferior, limite_superior))

        if valor_ajustado != valor_original:
            ativo_info = f" para {ticker}" if ticker else ""
            print(f"[⚠️ Ajuste LSTM{ativo_info}] {valor_original:.2f} → {valor_ajustado:.2f}")

            if session is not None:
                session["ajuste_lstm"] = True

        return valor_ajustado

    except Exception as e:
        print(f"[🚨 ERRO] Ajuste LSTM falhou: {str(e)}")
        return valor_original
    # =============================================================================
# Criação do modelo LSTM para treinamento (usado em train_cripto.py)
# =============================================================================
import tensorflow as tf
Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense

def criar_modelo(janela: int, saida: int):
    modelo = Sequential()
    modelo.add(LSTM(50, return_sequences=True, input_shape=(janela, 1)))
    modelo.add(LSTM(50))
    modelo.add(Dense(saida))
    modelo.compile(optimizer='adam', loss='mse')
    return modelo