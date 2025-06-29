import os
import yfinance as yf
import requests
import pandas as pd
from logger import uso_logger
from utils.logger_eventos import registrar_evento_fallback

def obter_dados_com_fallback(
    ticker: str,
    intervalo: str = "45min",
    periodo: str = "5d",
    outputsize: int = 130,
    preferencia: str = "auto"
) -> tuple[pd.DataFrame, str | None, str | None, str | None]:

    def tentar_yfinance():
        try:
            mapping = {
                "15min": ("15m", "5d"),
                "30min": ("30m", "5d"),
                "45min": ("30m", "5d"),
                "1h":    ("1h", "7d"),
                "2h":    ("1h", "14d"),
                "6h":    ("1h", "30d"),
                "1d":    ("1d", "6mo"),
                "5d":    ("1d", "1y"),
                "1m":    ("1d", "1y"),
            }
            yf_int, yf_per = mapping.get(intervalo, ("1d", "1mo"))
            df = yf.download(ticker, interval=yf_int, period=yf_per, auto_adjust=True, progress=False)
            if df.empty:
                raise RuntimeError("Sem dados do yfinance")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(col).strip() for col in df.columns]

            df.rename(columns=lambda x: x.strip().title().replace("Adj Close", "Close"), inplace=True)

            for col in ["Open", "High", "Low", "Close"]:
                if col not in df.columns:
                    raise RuntimeError(f"yfinance sem coluna {col}")
            if "Volume" not in df.columns:
                df["Volume"] = pd.NA

            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df = df.astype({"Open": "float", "High": "float", "Low": "float", "Close": "float"}, errors="ignore")
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
            df = df.ffill().dropna(subset=["Close"])

            uso_logger.info(f"[{ticker}] Dados obtidos via yfinance: {len(df)} linhas")
            return df[["Open", "High", "Low", "Close", "Volume"]], "yfinance", yf_int, None
        except Exception as e:
            uso_logger.warning(f"⚠️ yfinance falhou para {ticker}: {e}")
            return pd.DataFrame(), "yfinance", None, str(e)

    def tentar_twelvedata():
        apikey = os.getenv("TWELVE_DATA_API_KEY")
        if not apikey:
            err = "API key Twelve Data não encontrada"
            uso_logger.error(f"❌ {err}")
            return pd.DataFrame(), "twelvedata", None, err

        ticker_td = ticker.replace("-USD", "/USD").upper()
        tentativas = list(dict.fromkeys([intervalo, "15min", "30min", "1h", "2h", "4h", "1day"]))

        for alt in tentativas:
            try:
                url = "https://api.twelvedata.com/time_series"
                params = {
                    "symbol": ticker_td,
                    "interval": alt,
                    "outputsize": outputsize,
                    "apikey": apikey
                }
                resp = requests.get(url, params=params)
                resp.raise_for_status()
                js = resp.json()

                if "values" not in js or not js["values"]:
                    uso_logger.warning(f"[{ticker_td}] sem valores em {alt}")
                    continue

                df = pd.DataFrame(js["values"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                df.set_index("datetime", inplace=True)
                df.sort_index(inplace=True)

                keep = ["open", "high", "low", "close", "price", "volume"]
                cols = [c for c in keep if c in df.columns]
                df = df[cols]

                if "close" not in df.columns:
                    if "price" in df.columns:
                        df["close"] = pd.to_numeric(df["price"], errors="coerce")
                    else:
                        df["close"] = (pd.to_numeric(df["high"], errors="coerce") +
                                       pd.to_numeric(df["low"], errors="coerce")) / 2

                df = df.rename(columns={
                    "open": "Open", "high": "High",
                    "low": "Low", "close": "Close", "volume": "Volume"
                })
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col not in df.columns:
                        df[col] = pd.NA

                df = df[["Open", "High", "Low", "Close", "Volume"]]
                df = df.astype({"Open": "float", "High": "float", "Low": "float", "Close": "float"}, errors="ignore")
                df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
                df = df.ffill().dropna(subset=["Close"])

                msg = None
                if alt != intervalo:
                    msg = f"⚠️ Intervalo ajustado automaticamente para {alt}"
                    registrar_evento_fallback(ticker_td, alt, msg)

                uso_logger.info(f"[{ticker_td}] Dados obtidos via Twelve Data: {len(df)} linhas em {alt}")
                return df, "twelvedata", alt, msg

            except Exception as e:
                uso_logger.error(f"❌ erro Twelve Data [{alt}]: {e}")
                continue

        err = f"❌ não foi possível obter dados para {ticker}"
        uso_logger.error(err)
        return pd.DataFrame(), None, None, err

    # 🔁 Lógica de preferência
    if preferencia == "twelve":
        uso_logger.info(f"[{ticker}] Preferência forçada: Twelve Data.")
        return tentar_twelvedata()

    elif preferencia == "yahoo":
        retorno = tentar_yfinance()
        if retorno and not retorno[0].empty:
            return retorno
        return tentar_twelvedata()

    else:  # auto (padrão): prioriza Twelve Data para relatórios ao vivo
        retorno = tentar_twelvedata()
        if retorno and not retorno[0].empty:
            return retorno
        return tentar_yfinance()

def obter_preco_atual_binance(symbol: str = "PENDLEUSDT") -> float:
    """
    Retorna o último preço negociado via REST da Binance.
    Exemplo de symbol: 'BTCUSDT', 'ETHUSDT', 'PENDLEUSDT'
    """
    try:
        url = f"https://api.binance.com/api/v3/ticker/price"
        response = requests.get(url, params={"symbol": symbol.upper()}, timeout=5)
        response.raise_for_status()
        data = response.json()
        preco = float(data["price"])
        uso_logger.info(f"[Binance] Preço atual {symbol}: {preco:.4f}")
        return preco
    except Exception as e:
        uso_logger.warning(f"[Binance] Erro ao obter preço de {symbol}: {e}")
        return 0.0