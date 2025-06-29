import yfinance as yf

def carregar_dados(self):
    if "-USD" in self.ticker:
        df = yf.download(self.ticker, period="6mo", interval="1d", progress=False)
    else:
        df = yf.download(f"{self.ticker}.SA", period="6mo", interval="1d", progress=False)
        if df.empty or df['Close'].isna().all():
            # Fallback: tenta com o ticker puro
            df = yf.download(self.ticker, period="6mo", interval="1d", progress=False)

    if df.empty or "Close" not in df.columns:
        raise ValueError(f"Não foi possível carregar dados para o ticker '{self.ticker}'.")

    close = df['Close'].dropna().values
    if len(close) < self.janela + 1:
        raise ValueError(f"Dados insuficientes para treinar o modelo de {self.ticker}.")

    self.dados_treinamento = close