#!/usr/bin/env python3
import logging
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from utils.dados_com_fallback import obter_dados_com_fallback
from utils.dados_utils import preparar_dados_prophet
from utils.forecast_evaluation import backtest_evaluate, cv_summary, residuals_diagnostics
from utils.cv_utils import gerar_janelas_cv
from prophet_forecaster import ajustar_previsao_com_bollinger
from utils.indicadores import calcular_indicadores

# 1) Silencia o DEBUG interno de CmdStanPy e Prophet
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# 2) Configura o log do seu script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def validar_parametros_prophet(
    df_prophet: pd.DataFrame,
    intervalo_real: str,
    parametros: list[float]
) -> dict[float, float]:
    """
    Roda cross‐validation em df_prophet para cada changepoint_prior_scale em `parametros`,
    adaptando initial/period/horizon à frequência real dos dados.
    Retorna {scale: rmse-medio}.
    """
    n = len(df_prophet)

    # converte intervalo_real em minutos
    if intervalo_real.endswith("min"):
        freq_min = int(intervalo_real.replace("min",""))
    elif intervalo_real.endswith("h"):
        freq_min = int(intervalo_real.replace("h","")) * 60
    else:
        freq_min = 24*60

    # pontos
    init_pts    = max(3, int(n * 0.6))
    horizon_pts = max(1, int(n * 0.1))
    period_pts  = max(1, int((n - init_pts - horizon_pts) / 3))

    # minutos totais
    init_min    = init_pts    * freq_min
    horizon_min = horizon_pts * freq_min
    period_min  = period_pts  * freq_min

    # gera strings para Prophet
    def to_str(m):
        if m % 1440 == 0: return f"{m//1440} days"
        if m %   60 == 0: return f"{m//60} hours"
        return f"{m} minutes"

    init_str    = to_str(init_min)
    period_str  = to_str(period_min)
    horizon_str = to_str(horizon_min)

    logging.info(f"🧪 CV tuning (n={n}, initial={init_str}, period={period_str}, horizon={horizon_str})")

    resultados = {}
    for scale in parametros:
        m = Prophet(changepoint_prior_scale=scale)
        m.fit(df_prophet)
        try:
            df_cv = cross_validation(m, initial=init_str, period=period_str, horizon=horizon_str)
            perf  = performance_metrics(df_cv, rolling_window=1)
            resultados[scale] = perf["rmse"].mean()
            logging.info(f"✔ scale={scale} RMSE={resultados[scale]:.2f}")
        except Exception as e:
            logging.warning(f"⚠ scale={scale} falhou: {e}")
            resultados[scale] = float("nan")

    return resultados

def executar_pipeline(
    ticker: str,
    periodo: str = "120d",
    dias: int = 5
) -> pd.DataFrame:
    # 1) dados + intervalo real
    df_raw, fonte, intervalo_real, msg = obter_dados_com_fallback(
        ticker=ticker,
        intervalo="1d",
        periodo=periodo,
        preferencia="yahoo"  # ✅ evita uso desnecessário da Twelve Data em testes
    )

    if df_raw.empty or "Close" not in df_raw.columns or df_raw["Close"].dropna().empty:
        raise RuntimeError(f"Erro ao obter dados para {ticker}: {msg or 'Dados ausentes ou inválidos.'}")

    # 2) prepara para Prophet
    df_prophet = preparar_dados_prophet(df_raw)
    
    # Determina unidade com base no intervalo real
    unidade = "minutes" if intervalo_real.endswith("min") else "hours" if intervalo_real.endswith("h") else "days"

    # Recalcula janelas
    init_str, period_str, horizon_str = gerar_janelas_cv(len(df_prophet), unidade)

    # 3) tuning via CV adaptado
    scales = [0.01, 0.05, 0.1, 0.2]
    resultados = validar_parametros_prophet(df_prophet, intervalo_real, scales)
    validos = {s: rmse for s, rmse in resultados.items() if not pd.isna(rmse)}
    best_scale = min(validos, key=validos.get) if validos else 0.05
    logging.info(f"✔️ Best changepoint_prior_scale = {best_scale}")

    # **Recalcula as mesmas janelas para o cv_summary**
    n = len(df_prophet)
    # (repete exatamente o mesmo cálculo de validar_parametros_prophet)
    if intervalo_real.endswith("min"):
        freq_min = int(intervalo_real.replace("min",""))
    elif intervalo_real.endswith("h"):
        freq_min = int(intervalo_real.replace("h","")) * 60
    else:
        freq_min = 24*60

    init_pts    = max(3, int(n * 0.6))
    horizon_pts = max(1, int(n * 0.1))
    period_pts  = max(1, int((n - init_pts - horizon_pts) / 3))
    init_min    = init_pts    * freq_min
    horizon_min = horizon_pts * freq_min
    period_min  = period_pts  * freq_min
    def to_str(m):
        if m % 1440 == 0: return f"{m//1440} days"
        if m %   60 == 0: return f"{m//60} hours"
        return f"{m} minutes"
    init_str    = to_str(init_min)
    period_str  = to_str(period_min)
    horizon_str = to_str(horizon_min)

    # 4) modelo Prophet com sazonalidade e regressors
    modelo = Prophet(
        changepoint_prior_scale=best_scale,
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False
    )

    # 🔁 Sazonalidade por intervalo
    if intervalo_real == "15min":
        modelo.add_seasonality(name='15min_cycle', period=0.0104, fourier_order=5)
    elif intervalo_real == "30min":
        modelo.add_seasonality(name='30min_cycle', period=0.0208, fourier_order=5)
    elif intervalo_real == "45min" or intervalo_real == "1h":
        modelo.add_seasonality(name='1h_cycle', period=0.0416, fourier_order=5)
    elif intervalo_real in ["1d", "2d", "3d", "5d"]:
        modelo.add_seasonality(name='daily_cycle', period=1, fourier_order=10)

    # 📊 Regressor opcional: volume
    if "Volume" in df_prophet.columns:
        df_prophet['Volume'] = df_prophet['Volume'].fillna(method='ffill')
        modelo.add_regressor('Volume')

    # ✅ Checagem de variação nos dados
    if df_prophet['y'].nunique() < 3:
        raise RuntimeError(f"⚠️ Dados de 'y' com variação insuficiente para {ticker}")

    # Ajuste
    modelo.fit(df_prophet)

    # ⚙️ Frequência
    if intervalo_real.endswith("min"):
        freq = intervalo_real.replace("min", "min")
    elif intervalo_real.endswith("h"):
        freq = intervalo_real.replace("h", "H")
    else:
        freq = "D"

    # 4a) back‐test
    metrics_bt, _ = backtest_evaluate(df_prophet, best_scale, test_frac=0.2, freq=freq)
    logging.info(f"Backtest metrics: {metrics_bt}")

    # ✅ Validação do RMSE
    desvio_y = df_prophet['y'].std()
    if desvio_y > 0 and metrics_bt.get("RMSE", 0) > 3 * desvio_y:
        raise RuntimeError(
            f"⚠️ RMSE alto ({metrics_bt['RMSE']:.2f}) para {ticker}, indicando possível erro no modelo ou dados inconsistentes"
        )

    import os
    import csv
    from datetime import datetime

    def salvar_metrica(ticker, rmse, mape, scale, inicio, fim):
        caminho = "resultados_prophet.csv"
    
        # Criar cabeçalho se o arquivo ainda não existir
        if not os.path.exists(caminho):
            with open(caminho, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["DataExecucao", "Ticker", "RMSE", "MAPE", "BestScale", "PrevisaoInicio", "PrevisaoFim"])

        # Adicionar nova linha
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

    # 4b) CV summary no modelo final (agora sim com init_str, period_str, horizon_str definidos)
    modelo = Prophet(changepoint_prior_scale=best_scale)
    modelo.fit(df_prophet)
    perf_cv = cv_summary(modelo, init_str, period_str, horizon_str)

    # 4c) resíduos
    residuals_diagnostics(modelo, df_prophet)

    print("🧪 metrics_bt:", metrics_bt)
    # 4d) previsões finais
    from utils.previsao_utils import preencher_volume_futuro
    
    futuro = modelo.make_future_dataframe(periods=dias, freq=freq)
    futuro = preencher_volume_futuro(df_prophet, futuro, metodo="auto")

    # Se o modelo tiver 'Volume' como regressor, preenche no futuro
    if "Volume" in df_prophet.columns:
        try:
            futuro["Volume"] = futuro["Volume"]  # já foi preenchido
        except Exception:
            futuro["Volume"] = df_prophet["Volume"].ffill().iloc[-1]

        volume_futuro = df_prophet["Volume"].ffill().iloc[-1]
        futuro["Volume"] = volume_futuro

    previsao = modelo.predict(futuro)
    previsao["Volume"] = futuro["Volume"].values
    nome_arquivo = f"previsoes_prophet/prophet_{ticker.replace('-', '').replace('/', '')}.csv"
    colunas_para_salvar = ["ds", "yhat", "yhat_lower", "yhat_upper", "Volume"]
    previsao[colunas_para_salvar].tail(dias).to_csv(nome_arquivo, index=False)

    # ✅ Checagem de robustez da previsão
    if previsao is None or previsao.empty or "yhat" not in previsao.columns:
        raise RuntimeError(f"⚠️ Previsão Prophet malformada ou ausente para {ticker}")

    inicio = previsao["ds"].min()
    fim = previsao["ds"].max()
    salvar_metrica(ticker, metrics_bt["RMSE"], metrics_bt["MAPE"], best_scale, inicio, fim)
    return previsao[["ds","yhat","yhat_lower","yhat_upper"]].tail(dias)

if __name__ == "__main__":
    import os

    tickers = ["BTC-USD", "RAY-USD", "SOL-USD", "PENDLE-USD"]
    periodo = "120d"
    dias = 5

    os.makedirs("previsoes_prophet", exist_ok=True)

    for ticker in tickers:
        print(f"\n📊 Rodando previsão Prophet para {ticker}")
        try:
            df_final = executar_pipeline(ticker, periodo=periodo, dias=dias)
            print(df_final)
            
            from utils.dados_com_fallback import obter_dados_com_fallback

            # Carrega novamente os dados reais usados para calcular os indicadores técnicos
            dados_raw, _, _, _ = obter_dados_com_fallback(ticker, intervalo="1d", periodo=periodo)
            indicadores_df = calcular_indicadores(dados_raw)

            # Aplica o ajuste com Bollinger
            df_final = ajustar_previsao_com_bollinger(df_final, indicadores_df)

            # Salvar CSV por ativo
            nome_arquivo = f"previsoes_prophet/prophet_{ticker.replace('-', '')}.csv"
            df_final.to_csv(nome_arquivo, index=False)
            print(f"✅ Previsão salva em: {nome_arquivo}")

            # Verifica se há NaNs
            if df_final['yhat'].isna().any():
                print("⚠️ Previsão gerada com NaNs — revisar modelo ou dados.")
            else:
                print("✅ Previsão gerada com sucesso.")

        except Exception as e:
            print(f"❌ Erro ao processar {ticker}: {e}")
