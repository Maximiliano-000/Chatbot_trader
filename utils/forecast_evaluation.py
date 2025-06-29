import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import logging
import os

logger = logging.getLogger(__name__)

def backtest_evaluate(df_prophet, changepoint_prior_scale, test_frac=0.2, freq="D"):
    """
    Separa os últimos test_frac% dos pontos para teste.
    Retorna (metrics_dict, df_preds_vs_true).
    """
    n = len(df_prophet)
    split = int(n * (1 - test_frac))
    train = df_prophet.iloc[:split]
    test  = df_prophet.iloc[split:]

    # treina
    from prophet import Prophet
    m = Prophet(changepoint_prior_scale=changepoint_prior_scale)
    m.fit(train)

    # faz previsões
    future = m.make_future_dataframe(periods=len(test), freq=freq)
    fcst   = m.predict(future).set_index('ds')
    
    # Garante que todas as datas existam ou serão NaN
    fcst_t = fcst['yhat'].reindex(test['ds'])

    # Junta e remove as que não têm previsão
    df = test.set_index('ds').join(fcst_t.rename('yhat'))
    df = df.dropna(subset=['yhat'])

    # calcula métricas
    mse  = mean_squared_error(df['y'], df['yhat'])
    mae  = mean_absolute_error(df['y'], df['yhat'])
    mape = mean_absolute_percentage_error(df['y'], df['yhat'])

    rmse = mse ** 0.5
    metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    logger.info(f"Backtest metrics: {metrics}")

    # plot
    plt.figure()
    plt.plot(df.index, df['y'], label='true')
    plt.plot(df.index, df['yhat'], label='pred')
    plt.title("Backtest: true vs pred")
    plt.legend()
    plt.show()

    return metrics, df


def cv_summary(
    model,
    initial: str,
    period: str,
    horizon: str
) -> pd.DataFrame:
    """
    Roda cross-validation no modelo completo usando janelas parametrizadas.
    Retorna o DataFrame de performance (metrics).
    """
    logger.info(f"🧪 CV summary (initial={initial}, period={period}, horizon={horizon})")
    try:
        df_cv = cross_validation(
            model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        perf = performance_metrics(df_cv, rolling_window=1)
        logger.info(f"✔️ CV summary RMSE médio = {perf['rmse'].mean():.2f}")
        plot_cross_validation_metric(df_cv, metric='mape')
        return perf
    except ValueError as e:
        logger.warning(f"⚠ CV summary pulado: {e}")
        return pd.DataFrame()


def residuals_diagnostics(model, df_prophet):
    """
    Plota resíduos (y − yhat) no treino para checar autocorrelação / distribution.
    """
    hist = model.predict(df_prophet)
    res  = df_prophet['y'].values - hist['yhat'].values

    plt.figure()
    plt.hist(res, bins=30)
    plt.title("Histogram of residuals")
    plt.show()

    plt.figure()
    plt.plot(df_prophet['ds'], res)
    plt.title("Residuals over time")
    plt.show()

    return res

from datetime import datetime

def avaliar_previsao_real(previsao_df: pd.DataFrame, preco_real_df: pd.DataFrame, ticker: str, intervalo: str, salvar_em='avaliacoes_prophet.csv'):
    """
    Compara previsão x realizado, calcula RMSE/MAPE e salva score histórico.
    """
    try:
        df = previsao_df.merge(preco_real_df, on='ds', how='inner')
        df['erro_abs'] = (df['yhat'] - df['preco_real']).abs()
        df['erro_pct'] = df['erro_abs'] / df['preco_real'] * 100
        rmse = ((df['yhat'] - df['preco_real']) ** 2).mean() ** 0.5
        mape = df['erro_pct'].mean()

        resultado = {
            "data_avaliacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "intervalo": intervalo,
            "amostras": len(df),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 2)
        }

        df_resultado = pd.DataFrame([resultado])
        if not os.path.exists(salvar_em):
            df_resultado.to_csv(salvar_em, index=False)
        else:
            df_existente = pd.read_csv(salvar_em)
            df_completo = pd.concat([df_existente, df_resultado], ignore_index=True)
            df_completo.to_csv(salvar_em, index=False)

        logger.info(f"Avaliação registrada com sucesso: {resultado}")
        return resultado

    except Exception as e:
        logger.error(f"Erro ao avaliar previsão real: {e}")
        return None

def avaliar_lstm_vs_real(previsao_lstm: list[float], precos_reais: list[float], timestamps: list[str], ticker: str, intervalo: str, salvar_em='avaliacoes_lstm.csv'):
    df = pd.DataFrame({
        'ds': timestamps,
        'previsto_lstm': previsao_lstm,
        'preco_real': precos_reais
    })
    df['erro_abs'] = (df['previsto_lstm'] - df['preco_real']).abs()
    df['erro_pct'] = df['erro_abs'] / df['preco_real'] * 100
    rmse = ((df['previsto_lstm'] - df['preco_real']) ** 2).mean() ** 0.5
    mape = df['erro_pct'].mean()

    resultado = {
        "data_avaliacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "intervalo": intervalo,
        "amostras": len(df),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2)
    }

    df_resultado = pd.DataFrame([resultado])
    if not os.path.exists(salvar_em):
        df_resultado.to_csv(salvar_em, index=False)
    else:
        df_existente = pd.read_csv(salvar_em)
        df_completo = pd.concat([df_existente, df_resultado], ignore_index=True)
        df_completo.to_csv(salvar_em, index=False)

    logger.info(f"Avaliação LSTM registrada: {resultado}")
    return resultado

def avaliar_gatilho_atingido(preco_real: pd.Series, gatilho: float, alvo: float, tipo: str) -> bool:
    if tipo == 'alta' and preco_real.max() >= alvo and preco_real.min() >= gatilho:
        return True
    elif tipo == 'baixa' and preco_real.min() <= alvo and preco_real.max() <= gatilho:
        return True
    return False

def avaliar_rsi_comportamento(rsi_series, preco_series):
    zonas_sobrecompra = rsi_series > 70
    quedas_apos_pico = preco_series.diff().fillna(0) < 0
    acertos = (zonas_sobrecompra & quedas_apos_pico).sum()
    return acertos
