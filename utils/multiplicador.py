def obter_multiplicador_atr(ticker):
    """
    Retorna o multiplicador de ATR baseado no ativo.
    Criptos voláteis recebem multiplicador maior.
    """
    ticker = ticker.upper()

    if any(cripto in ticker for cripto in ["SOL", "PENDLE", "DOGE", "AVAX", "SAND", "SHIB"]):
        multiplicador = 2.0  # Criptomoedas muito voláteis
    elif any(moeda in ticker for moeda in ["BTC", "ETH", "BNB", "ADA"]):
        multiplicador = 1.5  # Criptomoedas grandes
    else:
        multiplicador = 1.2  # Ações ou ativos tradicionais

    print(f"[ATR] Multiplicador aplicado para {ticker}: {multiplicador}")
    return multiplicador