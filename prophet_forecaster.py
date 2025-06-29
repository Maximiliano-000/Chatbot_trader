import os
import csv
import logging
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from datetime import datetime
from utils.dados_utils import preparar_dados_prophet
from utils.previsao_utils import preencher_volume_futuro
from utils.forecast_evaluation import residuals_diagnostics, cv_summary, backtest_evaluate
from utils.indicadores import calcular_indicadores
from sklearn.metrics import mean_absolute_error

def ajustar_changepoint_dinamico(df, escalas=[0.01, 0.05, 0.1, 0.15]):
    melhores_metricas = []
    for esc in escalas:
        try:
            m = Prophet(
                changepoint_prior_scale=esc,
                seasonality_mode='multiplicative'
            )
            df['cap'] = df['y'].max() * 1.1
            df['floor'] = df['y'].min() * 0.9
            m.fit(df)
            previsto = m.predict(df)
            erro = mean_absolute_error(df['y'], previsto['yhat'])
            melhores_metricas.append((esc, erro))
        except Exception as e:
            logging.warning(f"Erro ao testar escala {esc}: {e}")
    if not melhores_metricas:
        raise RuntimeError("Nenhuma escala válida encontrada para o modelo.")
    melhor_escala = min(melhores_metricas, key=lambda x: x[1])[0]
    return melhor_escala

def ajustar_previsao_com_bollinger(previsao_df, indicadores_df, margem_pct=0.5):
    """
    Ajusta a previsão Prophet para se manter dentro de uma margem técnica segura:
    - Evita extrapolar Bollinger Bands (com margem de segurança percentual)
    - Aproxima da SMA20 se estiver fora das zonas técnicas
    """
    if not {'yhat', 'ds'}.issubset(previsao_df.columns) or indicadores_df.empty:
        return previsao_df

    try:
        ultima_sma = indicadores_df['SMA20'].iloc[-1]
        upper = indicadores_df['UpperBand'].iloc[-1]
        lower = indicadores_df['LowerBand'].iloc[-1]

        # Define margens ajustáveis
        faixa_superior = upper + (upper - ultima_sma) * margem_pct
        faixa_inferior = lower - (ultima_sma - lower) * margem_pct

        def ajustar(valor):
            valor_ajustado = min(max(valor, faixa_inferior), faixa_superior)
            return round(valor_ajustado, 4)

        previsao_df['yhat'] = previsao_df['yhat'].apply(ajustar)
        return previsao_df

    except Exception as e:
        logging.warning(f"⚠️ Erro no ajuste com Bollinger: {e}")
        return previsao_df

def executar_pipeline_completo(ticker: str, dados: pd.DataFrame, dias: int = 5, changepoint_scale: float = 0.05, freq: str = 'D') -> pd.DataFrame:
    """
    Executa a pipeline Prophet com validação, ajuste técnico com Bollinger, regressão de volume, e salva CSV + métricas.
    """
    try:
        df_prophet = preparar_dados_prophet(dados)
        if df_prophet.empty or df_prophet['y'].nunique() < 3 or len(df_prophet) < 10:
            raise ValueError("Dados insuficientes para o Prophet após preparação.")
    except Exception as e:
        logging.error(f"❌ Erro ao preparar dados para Prophet: {e}")
        raise RuntimeError(f"Dados insuficientes para o ativo {ticker}.")

    # Ajuste dinâmico do changepoint_prior_scale
    changepoint_scale = ajustar_changepoint_dinamico(df_prophet)

    # Define sazonalidades padrão
    sazonal_diaria = freq not in ["15min", "30min", "1h"]
    sazonal_semanal = True
    logging.info(f"Melhor escala selecionada: {changepoint_scale}")

    modelo = Prophet(
        changepoint_prior_scale=changepoint_scale,
        seasonality_mode='multiplicative',
        daily_seasonality=sazonal_diaria,
        weekly_seasonality=sazonal_semanal,
        yearly_seasonality=False
    )

    if 'Volume' in df_prophet.columns:
        modelo.add_regressor('Volume')

    modelo.fit(df_prophet)

    # Validações
    try:
        metrics_bt, _ = backtest_evaluate(df_prophet, changepoint_scale, test_frac=0.2, freq=freq)
        logging.info(f"Backtest metrics: {metrics_bt}")
    except Exception as e:
        logging.warning(f"⚠️ Erro ao calcular métricas Prophet: {e}")
        metrics_bt = {"RMSE": 0, "MAPE": 0}

    try:
        cv_summary(modelo, initial='60 days', period='15 days', horizon='5 days')
    except Exception as e:
        logging.warning(f"⚠️ CV Prophet falhou: {e}")

    residuals_diagnostics(modelo, df_prophet)

    # Geração futura
    futuro = modelo.make_future_dataframe(periods=dias, freq=freq)
    if futuro.empty or len(futuro) < dias:
        raise RuntimeError("❌ DataFrame futuro inválido.")

    futuro = preencher_volume_futuro(df_prophet, futuro)
    if "Volume" in futuro.columns and futuro["Volume"].isnull().all():
        raise RuntimeError("❌ Volume futuro ausente.")

    previsao = modelo.predict(futuro)
    if previsao.empty or "yhat" not in previsao.columns:
        raise RuntimeError(f"❌ Previsão Prophet malformada para {ticker}.")

    # Merge seguro sem perder índice nem gerar desalinhamento
    previsao = previsao.merge(futuro[['ds', 'Volume']], on='ds', how='left')

    # ✅ Ajuste técnico com Bollinger
    indicadores = calcular_indicadores(dados)
    previsao = ajustar_previsao_com_bollinger(previsao, indicadores)

    colunas_para_salvar = ["ds", "yhat", "yhat_lower", "yhat_upper", "Volume"]
    previsao_tail = previsao.tail(dias).copy().reset_index(drop=True)

    # Validação de segurança robusta
    for col in colunas_para_salvar:
        if col not in previsao_tail:
            previsao_tail[col] = pd.NA

    # Monte DataFrame seguro para exportação
    df_exportar = previsao_tail[colunas_para_salvar]

    # Salve CSV
    os.makedirs("previsoes_prophet", exist_ok=True)
    nome_arquivo = f"previsoes_prophet/prophet_{ticker.replace('-', '').replace('/', '')}.csv"
    df_exportar.to_csv(nome_arquivo, index=False)

    # Salve métrica
    salvar_metrica(
        ticker,
        metrics_bt["RMSE"],
        metrics_bt["MAPE"],
        changepoint_scale,
        df_exportar["ds"].min(),
        df_exportar["ds"].max()
    )

    # ✅ Retorne DataFrame seguro para o main.py (sem desalinhamentos!)
    return df_exportar[["ds", "yhat", "yhat_lower", "yhat_upper"]]

def salvar_metrica(ticker, rmse, mape, scale, inicio, fim):
    caminho = "resultados_prophet.csv"
    if not os.path.exists(caminho):
        with open(caminho, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["DataExecucao", "Ticker", "RMSE", "MAPE", "BestScale", "PrevisaoInicio", "PrevisaoFim"])

    with open(caminho, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticker,
            round(rmse, 4),
            round(mape, 6),
            scale,
            inicio.strftime("%Y-%m-%d %H:%M:%S"),
            fim.strftime("%Y-%m-%d %H:%M:%S")
        ])