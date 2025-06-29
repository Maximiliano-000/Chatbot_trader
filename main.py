
import pandas as pd
from prophet_forecaster import executar_pipeline_completo
from avaliador_completo import executar_avaliacao_completa
from binance.client import Client
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# === Configurações ===
ticker = "pendle-USD"
intervalo = "30min"
dias_previsao = 5
frequencia = "30min"

# === Caminhos ===
nome_limpo = ticker.replace("-", "")
caminho_dados = f"dados/precos_{nome_limpo}.csv"
caminho_prev_prophet = f"previsoes_prophet/prophet_{nome_limpo}.csv"
caminho_prev_lstm = f"avaliacoes/lstm_{ticker}.csv"
caminho_real = f"avaliacoes/precos_reais_{ticker}.csv"

# === Verificação de existência e consistência do arquivo CSV ===
if not os.path.exists(caminho_dados):
    arquivos_dados = os.listdir("dados")
    print(f"❌ Arquivo esperado não encontrado: {caminho_dados}")
    print("📂 Arquivos disponíveis em /dados:")
    for arq in arquivos_dados:
        print("  -", arq)

    sugestoes = [a for a in arquivos_dados if a.lower().replace('_', '') == os.path.basename(caminho_dados).lower().replace('_', '')]
    if sugestoes:
        print("💡 Sugestão: renomear o arquivo abaixo para o nome esperado:")
        print(f"mv dados/{sugestoes[0]} {caminho_dados}")
    else:
        print("⚠️ Nenhum arquivo similar encontrado. Verifique se o CSV foi gerado corretamente.")
    exit()

# === 1. Carregar os dados históricos ===
try:
    dados = pd.read_csv(caminho_dados)
except Exception as e:
    print(f"Erro ao carregar dados: {e}")
    exit()

# === 2. Gerar previsão Prophet ===
try:
    previsao_df = executar_pipeline_completo(ticker, dados, dias=dias_previsao, freq=frequencia)
    print(f"✅ Previsão Prophet gerada para {ticker}")
except Exception as e:
    print(f"[Erro] na geração da previsão Prophet: {e}")
    exit()

# (Opcional) salvar a previsão gerada
os.makedirs("previsoes_prophet", exist_ok=True)
previsao_df.to_csv(caminho_prev_prophet, index=False)

# === 3. Obter dados reais da Binance ===
def obter_precos_reais_binance(ticker_binance: str, intervalo_binance: str = "30m", limite: int = 100):
    client = Client()
    klines = client.get_klines(symbol=ticker_binance, interval=intervalo_binance, limit=limite)
    dados = []
    for linha in klines:
        timestamp = pd.to_datetime(linha[0], unit='ms')
        preco_fechamento = float(linha[4])
        dados.append({"ds": timestamp, "preco_real": preco_fechamento})
    return pd.DataFrame(dados)

ticker_binance = ticker.upper().replace("-", "") + "T"
real = obter_precos_reais_binance(ticker_binance=ticker_binance, intervalo_binance="30m", limite=50)
real = real.merge(previsao_df[["ds"]], on="ds", how="inner")  # alinha com previsão

# Verificação adicional
if real.empty:
    print("⚠️ Nenhum dado real compatível com a previsão. Avaliação abortada.")
    exit()

# Print de datas comparadas
print("🔎 Datas previstas:", previsao_df["ds"].dt.strftime("%Y-%m-%d %H:%M").tolist())
print("🔎 Datas reais coletadas:", real["ds"].dt.strftime("%Y-%m-%d %H:%M").tolist())

# Salvar real
os.makedirs("avaliacoes", exist_ok=True)
real.to_csv(caminho_real, index=False)

# === 4. Simular previsão LSTM e salvar ===
prev_lstm = real.copy()
prev_lstm["previsto_lstm"] = prev_lstm["preco_real"] * 0.997
prev_lstm.to_csv(caminho_prev_lstm, index=False)

# === 5. RSI e preços simulados ===
rsi_series = pd.Series([71, 73, 68, 65])
preco_series_rsi = pd.Series([4.50, 4.48, 4.45, 4.43])

# === 6. Avaliação completa ===
resultado = executar_avaliacao_completa(
    ticker=ticker,
    intervalo=intervalo,
    caminho_prev_prophet=caminho_prev_prophet,
    caminho_prev_lstm=caminho_prev_lstm,
    caminho_real=caminho_real,
    rsi_series=rsi_series,
    preco_series_rsi=preco_series_rsi,
    gatilho=4.49,
    alvo=4.64,
    tipo='alta'
)

print("🧾 Resultado da Avaliação:")
print(resultado)