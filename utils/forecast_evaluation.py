import os
import logging
from datetime import datetime

# =========================
# Headless / Backend setup
# =========================
HEADLESS = os.getenv("ANALIZ_HEADLESS", "1") == "1"
PLOTS_DIR = os.getenv("ANALIZ_PLOTS_DIR", "avaliacoes/plots")

if HEADLESS:
    import matplotlib
    matplotlib.use("Agg")  # precisa ser antes do pyplot

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from .metrics import directional_accuracy
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.diagnostics import cross_validation
logger = logging.getLogger(__name__)


# =========================
# Helpers de plot
# =========================
def _ensure_plots_dir():
    try:
        os.makedirs(PLOTS_DIR, exist_ok=True)
    except Exception as e:
        logger.warning(f"Não foi possível criar diretório de plots '{PLOTS_DIR}': {e}")

def _ts_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

def _maybe_show_or_save(fig=None, name_prefix: str = "plot"):
    """
    Se HEADLESS, salva PNG em avaliacoes/plots; senão, exibe.
    """
    if fig is None:
        fig = plt.gcf()
    if HEADLESS:
        _ensure_plots_dir()
        fname = os.path.join(PLOTS_DIR, _ts_name(name_prefix) + ".png")
        try:
            fig.savefig(fname, dpi=120, bbox_inches="tight")
            logger.info(f"Figura salva: {fname}")
        except Exception as e:
            logger.warning(f"Falha ao salvar figura: {e}")
        finally:
            plt.close(fig)
    else:
        plt.show()


# =========================
# Backtest (holdout simples)
# =========================
def backtest_evaluate(df_prophet: pd.DataFrame, changepoint_prior_scale, test_frac: float = 0.2, freq: str = "D"):
    """
    Separa os últimos test_frac% para teste.
    Retorna (metrics_dict, df_preds_vs_true).
    df_prophet: DataFrame com colunas ['ds', 'y']
    """
    n = len(df_prophet)
    if n < 10:
        logger.warning(f"Poucos dados para backtest: n={n}")
    split = int(n * (1 - test_frac))
    train = df_prophet.iloc[:split]
    test  = df_prophet.iloc[split:]

    from prophet import Prophet
    m = Prophet(changepoint_prior_scale=changepoint_prior_scale)
    m.fit(train)

    future = m.make_future_dataframe(periods=len(test), freq=freq)
    fcst   = m.predict(future).set_index('ds')

    # Reindexa pelo teste e remove NaNs
    fcst_t = fcst['yhat'].reindex(test['ds'])
    df = test.set_index('ds').join(fcst_t.rename('yhat'))
    df = df.dropna(subset=['yhat'])

    # Métricas
    mse  = mean_squared_error(df['y'], df['yhat'])
    mae  = mean_absolute_error(df['y'], df['yhat'])
    mape = mean_absolute_percentage_error(df['y'], df['yhat'])
    rmse = mse ** 0.5
    metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    logger.info(f"Backtest metrics: {metrics}")

    # Plot
    plt.figure()
    plt.plot(df.index, df['y'], label='true')
    plt.plot(df.index, df['yhat'], label='pred')
    plt.title("Backtest: true vs pred")
    plt.legend()
    _maybe_show_or_save(name_prefix="backtest_true_vs_pred")

    return metrics, df


# --- helpers opcionais (podem ficar no topo do arquivo) ---
def _parse_int(s: str) -> int:
    try:
        return int((s or "0").split()[0])
    except Exception:
        return 0

def _parse_unit(s: str) -> str:
    s = (s or "").strip().lower()
    toks = s.split()
    return toks[-1] if toks else "days"


# =========================
# Cross-Validation (Prophet)
# =========================
def cv_summary(model, initial: str, period: str, horizon: str) -> pd.DataFrame:
    """
    Roda cross-validation no modelo completo usando janelas parametrizadas.
    Retorna o DataFrame de performance (metrics).
    Ex.: initial='365 days', period='30 days', horizon='90 days'
    """
    # --- ADAPTAÇÃO SEGURA (tudo dentro da função!) ---
    unit_h = _parse_unit(horizon)
    unit_i = _parse_unit(initial) or unit_h
    unit_p = _parse_unit(period)  or unit_h

    try:
        n_hist = len(getattr(model, "history", []))
    except Exception:
        n_hist = 0

    # horizon entre 5 e 10 “pontos” na unidade do horizon requisitado
    h_use = max(5, min(10, n_hist // 8)) if n_hist > 0 else 5
    horizon_safe = f"{h_use} {unit_h}"

    # regra prática: initial >= 3 * horizon
    ini_num = _parse_int(initial) or (3 * h_use)
    if ini_num < 3 * h_use:
        initial = f"{3 * h_use} {unit_i}"

    # garantir period <= horizon
    per_num = _parse_int(period) or h_use
    if per_num > h_use:
        period = f"{h_use} {unit_p}"

    logger.info(
        f"🧪 CV params ajustados: initial={initial}, period={period}, "
        f"horizon_req={horizon} -> horizon_used={horizon_safe}, n_hist={n_hist}"
    )

    # --- CHAMADA DO CV ---
    try:
        df_cv = cross_validation(
            model,
            initial=initial,
            period=period,
            horizon=horizon_safe
        )
        perf = performance_metrics(df_cv, rolling_window=1)
        logger.info(f"✔️ CV summary RMSE médio = {perf['rmse'].mean():.4f}")

        ax = plot_cross_validation_metric(df_cv, metric='mape')
        fig = ax.get_figure() if hasattr(ax, "get_figure") else plt.gcf()
        _maybe_show_or_save(fig, name_prefix="cv_mape")

        return perf
    except ValueError as e:
        logger.warning(f"⚠ CV summary pulado: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Erro inesperado no CV: {e}")
        return pd.DataFrame()



# =========================
# Diagnóstico de resíduos
# =========================
def residuals_diagnostics(model, df_prophet: pd.DataFrame):
    """
    Plota resíduos (y − yhat) no treino para checar autocorrelação / distribuição.
    """
    hist = model.predict(df_prophet)
    res  = df_prophet['y'].values - hist['yhat'].values

    plt.figure()
    plt.hist(res, bins=30)
    plt.title("Histogram of residuals")
    _maybe_show_or_save(name_prefix="resid_hist")

    plt.figure()
    plt.plot(df_prophet['ds'], res)
    plt.title("Residuals over time")
    _maybe_show_or_save(name_prefix="resid_series")

    return res


# =========================
# Avaliações (Prophet / LSTM)
# =========================
def avaliar_previsao_real(
    previsao_df: pd.DataFrame,
    preco_real_df: pd.DataFrame,
    ticker: str,
    intervalo: str,
    salvar_em: str = 'avaliacoes_prophet.csv'
):
    """
    Compara previsão x realizado, calcula RMSE/MAPE/DA e salva score histórico.
    """
    try:
        df = previsao_df.merge(preco_real_df, on='ds', how='inner')
        df['erro_abs'] = (df['yhat'] - df['preco_real']).abs()
        df['erro_pct'] = df['erro_abs'] / df['preco_real'] * 100
        rmse = ((df['yhat'] - df['preco_real']) ** 2).mean() ** 0.5
        mape = df['erro_pct'].mean()

        # Directional Accuracy (DA)
        da = directional_accuracy(df['preco_real'].values, df['yhat'].values)

        resultado = {
            "data_avaliacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "intervalo": intervalo,
            "amostras": len(df),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 2),
            "DA": round(da, 3),
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


def avaliar_lstm_vs_real(
    previsao_lstm: list[float],
    precos_reais: list[float],
    timestamps: list[str],
    ticker: str,
    intervalo: str,
    salvar_em: str = 'avaliacoes_lstm.csv'
):
    df = pd.DataFrame({
        'ds': timestamps,
        'previsto_lstm': previsao_lstm,
        'preco_real': precos_reais
    })
    df['erro_abs'] = (df['previsto_lstm'] - df['preco_real']).abs()
    df['erro_pct'] = df['erro_abs'] / df['preco_real'] * 100
    rmse = ((df['previsto_lstm'] - df['preco_real']) ** 2).mean() ** 0.5
    mape = df['erro_pct'].mean()

    # Directional Accuracy (DA)
    da = directional_accuracy(df['preco_real'].values, df['previsto_lstm'].values)

    resultado = {
        "data_avaliacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "intervalo": intervalo,
        "amostras": len(df),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2),
        "DA": round(da, 3),
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


# =========================
# Utilitários de avaliação extra
# =========================
def avaliar_gatilho_atingido(preco_real: pd.Series, gatilho: float, alvo: float, tipo: str) -> bool:
    if tipo == 'alta' and preco_real.max() >= alvo and preco_real.min() >= gatilho:
        return True
    elif tipo == 'baixa' and preco_real.min() <= alvo and preco_real.max() <= gatilho:
        return True
    return False

def avaliar_rsi_comportamento(rsi_series: pd.Series, preco_series: pd.Series):
    zonas_sobrecompra = rsi_series > 70
    quedas_apos_pico = preco_series.diff().fillna(0) < 0
    acertos = (zonas_sobrecompra & quedas_apos_pico).sum()
    return acertos