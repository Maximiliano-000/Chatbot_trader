import os
import requests
import pandas as pd
from logger import uso_logger

# Função existente, robusta com Twelve Data
def obter_dados(ticker, intervalo="1day", outputsize=130):
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        uso_logger.error("❌ TWELVE_DATA_API_KEY não encontrado no .env.")
        return pd.DataFrame()

    ticker_td = ticker.replace("-USD", "/USD").replace(".SA", "").upper()
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": ticker_td,
        "interval": intervalo,
        "outputsize": outputsize,
        "apikey": api_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        json_data = response.json()

        if "values" not in json_data:
            uso_logger.error(f"⚠️ Twelve Data sem valores para {ticker_td}: {json_data.get('message', 'Mensagem indisponível')}")
            return pd.DataFrame()

        df = pd.DataFrame(json_data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()

        df = df.rename(columns={k: k.lower() for k in df.columns})
        if "close" not in df.columns and "price" in df.columns:
            df["close"] = df["price"]
        if "volume" not in df.columns:
            df["volume"] = pd.NA

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                uso_logger.warning(f"⚠️ Coluna '{col}' ausente em {ticker_td}.")
                df[col] = pd.NA
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        }, inplace=True)

        df = df[["Open", "High", "Low", "Close", "Volume"]]

        if df[["High", "Low", "Close"]].isnull().all().any():
            uso_logger.error(f"❌ Dados técnicos incompletos para {ticker_td}.")
            return pd.DataFrame()

        uso_logger.info(f"[{ticker_td}] Dados carregados corretamente do Twelve Data.")
        return df

    except requests.exceptions.Timeout:
        uso_logger.error(f"⏱️ Timeout ao obter dados para {ticker_td} no intervalo {intervalo}")
        return pd.DataFrame()

    except Exception as e:
        uso_logger.error(f"⚠️ Erro ao acessar Twelve Data para {ticker_td}: {str(e)}")
        return pd.DataFrame()

# Nova função robusta e alternativa usando Binance API
def obter_dados_binance(symbol='SOLUSDT', interval='45m', limit=130):
    try:
        url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data, columns=[
            'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
            'CloseTime', 'QuoteVolume', 'Trades', 'TakerBaseVol', 'TakerQuoteVol', 'Ignore'
        ])

        df['datetime'] = pd.to_datetime(df['OpenTime'], unit='ms')
        df.set_index('datetime', inplace=True)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

        uso_logger.info(f"[Binance {symbol}] Dados carregados com sucesso.")
        return df

    except requests.exceptions.Timeout:
        uso_logger.error(f"⏱️ Timeout ao obter dados da Binance para {symbol} no intervalo {interval}")
        return pd.DataFrame()

    except Exception as e:
        uso_logger.error(f"⚠️ Erro ao acessar Binance API para {symbol}: {str(e)}")
        return pd.DataFrame()