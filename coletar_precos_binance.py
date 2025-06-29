from binance.client import Client
import pandas as pd
import os

symbol = "PENDLEUSDT"
intervalo = "30m"
limite = 100
caminho_arquivo = "avaliacoes/precos_reais_pendle-USD.csv"
os.makedirs("avaliacoes", exist_ok=True)

try:
    client = Client()
    klines = client.get_klines(symbol=symbol, interval=intervalo, limit=limite)
except Exception as e:
    print("❌ Erro ao conectar com Binance:", e)
    exit()

dados = []
for k in klines:
    dados.append({
        "ds": pd.to_datetime(k[0], unit='ms'),
        "preco_real": float(k[4])  # fechamento
    })

df = pd.DataFrame(dados)
print(f"✅ Linhas coletadas: {len(df)}")

if df.empty:
    print("⚠️ Nenhum dado retornado. Verifique se o ativo existe e está com negociação ativa.")
    exit()

try:
    df.to_csv(caminho_arquivo, index=False)
    print("✅ Arquivo salvo com sucesso:", os.path.abspath(caminho_arquivo))
except Exception as e:
    print("❌ Erro ao salvar CSV:", e)