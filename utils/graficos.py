import matplotlib
import matplotlib.pyplot as plt
import base64
import pandas as pd
import numpy as np
from io import BytesIO
from prophet import Prophet
from logger import uso_logger

# Corrige erro 'glyf' no PDF e padroniza visual
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# =============================
# 1. Gerar gráfico (HTML e PDF)
# =============================
"""
    Gera um gráfico técnico avançado em formato HTML ou PDF, contendo preço, 
    médias móveis, Bandas de Bollinger, RSI extremos, cruzamentos e rompimentos.
"""

def gerar_grafico(df, ticker, modo='html'):
    colunas_necessarias = ['Close', 'SMA20', 'SMA50', 'UpperBand', 'LowerBand', 'RSI']
    if not all(col in df.columns for col in colunas_necessarias):
        raise ValueError("Dados insuficientes para gerar o gráfico técnico.")

    df_plot = df[colunas_necessarias].copy().dropna()
    if df_plot.shape[0] < 2:
        raise ValueError("Dados insuficientes após limpeza para plotagem.")

    # Prophet opcional para changepoints
    changepoints = []
    if modo == 'completo':
        try:
            df_prophet = df.reset_index().iloc[:, [0, df.columns.get_loc('Close') + 1]].dropna()
            df_prophet.columns = ['ds', 'y']
            if df_prophet.shape[0] >= 2:
                modelo = Prophet()
                modelo.fit(df_prophet)
                changepoints = modelo.changepoints
        except Exception as e:
            uso_logger.error(f"⚠️ Prophet falhou: {e}")

    # Plot principal
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df_plot.index, df_plot['Close'], label='Preço', color='deepskyblue')
    ax.plot(df_plot.index, df_plot['SMA20'], label='SMA20', color='orange')
    ax.plot(df_plot.index, df_plot['SMA50'], label='SMA50', color='limegreen')
    ax.plot(df_plot.index, df_plot['UpperBand'], '--', label='Banda Superior', color='magenta')
    ax.plot(df_plot.index, df_plot['LowerBand'], '--', label='Banda Inferior', color='violet')

    # Linha inicial do período
    ax.axvline(df_plot.index[0], color='gray', linestyle='--', linewidth=0.8, alpha=0.4, label='Início')

    # Changepoints Prophet
    for cp in changepoints:
        ax.axvline(cp.to_pydatetime(), color='gray', linestyle='--', linewidth=0.8, alpha=0.4)

    # RSI extremos
    extremos_rsi = df_plot[(df_plot['RSI'] < 30) | (df_plot['RSI'] > 70)]
    ax.scatter(extremos_rsi.index, extremos_rsi['Close'], color='red', marker='^', label='RSI Extremo')

    # Cruzamentos de médias móveis
    cross_up = (df_plot['SMA20'] > df_plot['SMA50']) & (df_plot['SMA20'].shift(1) <= df_plot['SMA50'].shift(1))
    cross_down = (df_plot['SMA20'] < df_plot['SMA50']) & (df_plot['SMA20'].shift(1) >= df_plot['SMA50'].shift(1))
    ax.scatter(df_plot.index[cross_up], df_plot['Close'][cross_up], color='yellow', marker='o', label='Cruzamento Altista')
    ax.scatter(df_plot.index[cross_down], df_plot['Close'][cross_down], color='white', marker='v', label='Cruzamento Baixista')

    # Rompimentos de Bollinger
    try:
        close, upper = df_plot['Close'], df_plot['UpperBand']
        close_aligned, upper_aligned = close.align(upper, join='inner')
        _, lower_aligned = close.align(df_plot['LowerBand'], join='inner')

        rompe_cima = close_aligned > upper_aligned
        rompe_baixo = close_aligned < lower_aligned

        ax.scatter(close_aligned.index[rompe_cima], close_aligned[rompe_cima], 
                    color='cyan', marker='s', label='Romp. Superior')
        ax.scatter(close_aligned.index[rompe_baixo], close_aligned[rompe_baixo], 
                    color='magenta', marker='x', label='Romp. Inferior')

    except Exception as e:
        uso_logger.error(f"⚠️ Erro ao marcar rompimentos: {e}")

    # Layout
    ax.set_title(f"{ticker} – Indicadores Técnicos Avançados")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço (R$)")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Exportação em base64
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    return f'<img src="data:image/png;base64,{img_base64}" style="width:100%;" />'