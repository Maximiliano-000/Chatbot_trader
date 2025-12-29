import numpy as np
import pandas as pd

# =============================
# 1) Indicadores técnicos
# =============================
"""
Calcula e retorna indicadores técnicos como:
- SMA20/SMA50
- Bandas de Bollinger
- RSI
- MACD/MACD_Signal
- Volume_Medio (com fallbacks)
"""

def calcular_indicadores(dados, intervalo: str = "1d") -> pd.DataFrame:
    df = dados.copy()
    df.index = pd.to_datetime(df.index)

    if "Close" not in df.columns:
        return pd.DataFrame()

    min_candles_por_intervalo = {
        "15min": 20,
        "30min": 25,
        "45min": 30,
        "1h": 35,
        "2h": 30,
        "6h": 30,
        "1d": 52,
    }
    min_candles = min_candles_por_intervalo.get(intervalo, 30)

    if len(df) < min_candles:
        print(f"[ERRO] Dados insuficientes ({len(df)} candles). Mínimo exigido: {min_candles}.")
        return pd.DataFrame()

    if df["Close"].isnull().all():
        print("[ERRO] Todos valores de 'Close' são NaN.")
        return pd.DataFrame()

    # --- Médias e Bollinger ---
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()

    df["STD20"] = df["Close"].rolling(window=20).std()
    df["UpperBand"] = df["SMA20"] + (2 * df["STD20"])
    df["LowerBand"] = df["SMA20"] - (2 * df["STD20"])

    # --- RSI (robusto a séries curtas) ---
    delta = df["Close"].diff()
    win = 14 if len(df) >= 14 else max(2, len(df) // 2 or 2)
    ganho = delta.clip(lower=0).rolling(window=win, min_periods=1).mean()
    perda = -delta.clip(upper=0).rolling(window=win, min_periods=1).mean()
    rs = ganho / (perda + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # --- Volume / Volume_Medio com fallbacks ---
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        # Evita zeros que poluem a média
        df.loc[df["Volume"] == 0, "Volume"] = np.nan

        if df["Volume"].notna().any():
            df["Volume_Medio"] = df["Volume"].rolling(window=21, min_periods=1).mean()
        else:
            print("[AVISO] Não há valores válidos para cálculo de Volume Médio. Definindo 0.")
            df["Volume"] = 0.0
            df["Volume_Medio"] = 0.0
    else:
        print("[AVISO] Coluna 'Volume' não encontrada. Definindo como 0.")
        df["Volume"] = 0.0
        df["Volume_Medio"] = 0.0

    # --- Tipagem numérica consistente ---
    colunas_float = [
        "Close",
        "SMA20",
        "SMA50",
        "UpperBand",
        "LowerBand",
        "RSI",
        "MACD",
        "MACD_Signal",
        "Volume_Medio",
    ]
    df[colunas_float] = df[colunas_float].apply(pd.to_numeric, errors="coerce")

    # Remove apenas linhas sem Close
    df = df.dropna(subset=["Close"])
    if df.empty:
        print("[ERRO] DataFrame vazio após remover 'Close' nulos.")
        return pd.DataFrame()

    # Interpola suavemente indicadores (NÃO inventa Volume_Medio)
    interp_cols = [c for c in colunas_float if c != "Volume_Medio"]
    df[interp_cols] = (
        df[interp_cols]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )
    df["Volume_Medio"] = df["Volume_Medio"].ffill().bfill()

    return df


def calcular_fibonacci(
    preco_min: float,
    preco_max: float,
    *,
    arredondar: bool = False,
    casas: int = 4,
    validar_range: bool = True,
    min_range_pct: float = 0.01
) -> dict:
    """
    Calcula níveis de retração de Fibonacci a partir de um range válido.

    - Não arredonda por padrão (precisão para decisão).
    - Valida range mínimo (evita ruído).
    - Arredondamento apenas opcional (apresentação).
    """

    if preco_min is None or preco_max is None:
        return {}

    preco_min = float(preco_min)
    preco_max = float(preco_max)

    if preco_max <= preco_min:
        return {}

    diff = preco_max - preco_min

    # valida se o range é relevante
    if validar_range:
        range_pct = diff / preco_max
        if range_pct < min_range_pct:
            return {}

    fib = {
        "0.0%": preco_max,
        "23.6%": preco_max - 0.236 * diff,
        "38.2%": preco_max - 0.382 * diff,
        "50.0%": preco_max - 0.5 * diff,
        "61.8%": preco_max - 0.618 * diff,
        "100.0%": preco_min,
    }

    if arredondar:
        fib = {k: round(v, casas) for k, v in fib.items()}

    return fib

# =============================
# 2) Estratégias (Long / Short)
# =============================

def calcular_estrategia_longa(
    preco_atual,
    atr=None,
    ticker=None,
    cenarios=None,
    previsao_lstm=None,
    previsao_prophet=None,
    rsi=None,
    candle_reversao=False,
    volume_crescente=False,
    suporte_fibo=False,
    cruzamento_macd=False,
    divergencia_rsi=False,
    sobrevenda=False,
):
    def avaliar_condicoes_estrategia_compra(
        rsi,
        candle_reversao,
        volume_crescente,
        suporte_fibo,
        cruzamento_macd,
        divergencia_rsi,
        sobrevenda,
    ):
        sinais_forca = any(
            [
                candle_reversao,
                volume_crescente,
                suporte_fibo,
                cruzamento_macd,
                divergencia_rsi,
            ]
        )

        if sobrevenda and suporte_fibo and sinais_forca:
            return {
                "tipo": "Long",
                "mensagem": "Compra sugerida com base em sobrevenda, suporte técnico e sinais confirmados de força compradora.",
            }
        elif sobrevenda and suporte_fibo:
            return {
                "tipo": "Observacao",
                "mensagem": "Zona de atenção técnica: sobrevenda e suporte identificados, mas sem sinais claros de força. Aguardar confirmação.",
            }
        else:
            return {
                "tipo": "Neutra",
                "mensagem": "Sem evidências suficientes para sugerir compra no momento.",
            }

    avaliacao = avaliar_condicoes_estrategia_compra(
        rsi,
        candle_reversao,
        volume_crescente,
        suporte_fibo,
        cruzamento_macd,
        divergencia_rsi,
        sobrevenda,
    )

    try:
        atr = float(atr) if isinstance(atr, (int, float, str)) else atr.get("valor", 0)
    except (TypeError, ValueError, AttributeError):
        atr = 0.0

    entrada = round(preco_atual, 2)
    from .multiplicador import obter_multiplicador_atr
    multiplicador = obter_multiplicador_atr(ticker) if ticker else 1.5

    # Alvos sugeridos por modelos
    alvos_modelos = []
    if previsao_lstm and len(previsao_lstm) > 0:
        alvos_modelos.append(max(previsao_lstm))
    if previsao_prophet is not None and "yhat_upper" in getattr(previsao_prophet, "columns", []):
        alvos_modelos.append(previsao_prophet["yhat_upper"].max())
    alvo_max_modelos = max(alvos_modelos) if alvos_modelos else preco_atual * 1.03

    # ATR limitado a 5% do preço
    atr_limitado = min(atr, preco_atual * 0.05)

    if atr_limitado > 0:
        tp1 = min(round(preco_atual + atr_limitado * 0.5, 2), alvo_max_modelos)
        tp2 = min(round(preco_atual + atr_limitado * 1.0, 2), alvo_max_modelos * 1.005)
        tp3 = min(round(preco_atual + atr_limitado * 1.5, 2), alvo_max_modelos * 1.010)
        sl = round(preco_atual - atr_limitado * multiplicador, 2)
    else:
        tp1, tp2, tp3 = [round(preco_atual * f, 2) for f in (1.02, 1.04, 1.06)]
        sl = round(preco_atual * 0.99, 2)

    tp1, tp2, tp3 = sorted([tp1, tp2, tp3])

    # Ajuste por cenários alternativos (se existirem)
    if cenarios and isinstance(cenarios, list):
        for c in cenarios:
            if "alta" in c.get("cenário", "").lower():
                try:
                    alvo_cenario = float(c["alvo"].replace("R$", "").replace(",", ".").strip())
                    gatilho_cenario = float(c["gatilho"].replace("R$", "").replace(",", ".").strip())
                    tp3 = max(tp3, alvo_cenario)
                    sl = min(sl, gatilho_cenario - 0.5)
                except Exception:
                    pass

    return {
        "avaliacao": avaliacao,
        "tipo": avaliacao["tipo"],
        "mensagem": avaliacao["mensagem"],
        "preco_entrada": entrada,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl": sl,
    }


def calcular_estrategia_short(
    preco_atual,
    atr=None,
    ticker=None,
    cenarios=None,
    previsao_lstm=None,
    previsao_prophet=None,
    rsi=None,
    candle_reversao=False,
    volume_decrescente=False,
    rejeicao_resistencia=False,
    cruzamento_macd=False,
    divergencia_rsi=False,
    sobrecompra=False,
    resistencia_fibo=False,
):
    def avaliar_condicoes_estrategia_venda(
        rsi,
        candle_reversao,
        volume_decrescente,
        rejeicao_resistencia,
        cruzamento_macd,
        divergencia_rsi,
        sobrecompra,
        resistencia_fibo,
    ):
        sinais_fraqueza = any(
            [
                candle_reversao,
                volume_decrescente,
                rejeicao_resistencia,
                cruzamento_macd,
                divergencia_rsi,
            ]
        )

        if sobrecompra and resistencia_fibo and sinais_fraqueza:
            return {
                "tipo": "Short",
                "mensagem": "Venda sugerida com base em sobrecompra, resistência e sinais confirmados de fraqueza (reversão ou exaustão).",
            }
        elif sobrecompra and resistencia_fibo:
            return {
                "tipo": "Observacao",
                "mensagem": "Zona de atenção técnica: sobrecompra e resistência detectadas, mas sem sinais claros de reversão. Aguardar confirmação.",
            }
        else:
            return {
                "tipo": "Neutra",
                "mensagem": "Sem evidências suficientes para sugerir venda no momento.",
            }

    avaliacao = avaliar_condicoes_estrategia_venda(
        rsi,
        candle_reversao,
        volume_decrescente,
        rejeicao_resistencia,
        cruzamento_macd,
        divergencia_rsi,
        sobrecompra,
        resistencia_fibo,
    )

    try:
        atr = float(atr) if isinstance(atr, (int, float, str)) else atr.get("valor", 0)
    except (TypeError, ValueError, AttributeError):
        atr = 0.0

    entrada = round(preco_atual, 2)
    from .multiplicador import obter_multiplicador_atr
    multiplicador = obter_multiplicador_atr(ticker) if ticker else 1.5

    # Alvos sugeridos por modelos
    alvos_modelos = []
    if previsao_lstm and len(previsao_lstm) > 0:
        alvos_modelos.append(min(previsao_lstm))
    if previsao_prophet is not None and "yhat_lower" in getattr(previsao_prophet, "columns", []):
        alvos_modelos.append(previsao_prophet["yhat_lower"].min())
    alvo_min_modelos = min(alvos_modelos) if alvos_modelos else preco_atual * 0.97

    # ATR limitado a 5% do preço
    atr_limitado = min(atr, preco_atual * 0.05)

    if atr_limitado > 0:
        tp1 = max(round(preco_atual - atr_limitado * 0.5, 2), alvo_min_modelos)
        tp2 = max(round(preco_atual - atr_limitado * 1.0, 2), alvo_min_modelos * 0.995)
        tp3 = max(round(preco_atual - atr_limitado * 1.5, 2), alvo_min_modelos * 0.990)
        sl = round(preco_atual + atr_limitado * multiplicador, 2)
    else:
        tp1, tp2, tp3 = [round(preco_atual * f, 2) for f in (0.98, 0.96, 0.94)]
        sl = round(preco_atual * 1.01, 2)

    tp1, tp2, tp3 = sorted([tp1, tp2, tp3], reverse=True)

    # Ajuste por cenários alternativos (se existirem)
    if cenarios and isinstance(cenarios, list):
        for c in cenarios:
            if "queda" in c.get("cenário", "").lower():
                try:
                    alvo_cenario = float(c["alvo"].replace("R$", "").replace(",", ".").strip())
                    gatilho_cenario = float(c["gatilho"].replace("R$", "").replace(",", ".").strip())
                    tp3 = min(tp3, alvo_cenario)
                    sl = gatilho_cenario + 0.5
                except Exception:
                    pass

    return {
        "avaliacao": avaliacao,
        "tipo": avaliacao["tipo"],
        "mensagem": avaliacao["mensagem"],
        "preco_entrada": entrada,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl": sl,
    }


# =============================
# 3) Microtendência e Reversões
# =============================

def gerar_microtendencia(preco_atual: float, previsoes_lstm: list) -> str:
    """
    Texto interpretativo com base nas próximas previsões LSTM e no preço atual.
    """
    if not previsoes_lstm or preco_atual is None or preco_atual < 0.01:
        return "⚠️ Dados insuficientes para estimar microtendência."

    try:
        valores = [float(p["valor"]) if isinstance(p, dict) else float(p) for p in previsoes_lstm[:3]]
    except Exception:
        return "⚠️ Erro ao processar as previsões LSTM."

    media_prevista = sum(valores) / len(valores)
    delta = round(media_prevista - preco_atual, 3)
    preco_formatado = f"{preco_atual:.2f}"
    direcao = "alta" if valores[0] < valores[-1] else "queda" if valores[0] > valores[-1] else "estavel"

    if abs(delta) < 0.01:
        if direcao == "alta":
            return f"📈 Tendência leve de alta em formação a partir de R$ {preco_formatado}."
        elif direcao == "queda":
            return f"📉 Tendência leve de baixa em formação a partir de R$ {preco_formatado}."
        else:
            return f"🔁 Estabilidade próxima em torno de R$ {preco_formatado} – sem movimento claro nos próximos candles."
    elif delta > 0:
        return f"📈 Leve alta projetada de até +{delta:.2f} a partir de R$ {preco_formatado} nos próximos candles."
    else:
        return f"📉 Leve queda projetada de até {abs(delta):.2f} a partir de R$ {preco_formatado} nos próximos candles."


def validar_reversao_baixa(indicadores: pd.DataFrame) -> bool:
    """
    Reversão de alta -> baixa:
    - pavio superior longo
    - fechamento abaixo da SMA20
    - volume decrescendo nos últimos 3 candles
    """
    req = {"Open", "High", "Low", "Close", "SMA20", "Volume"}
    if not isinstance(indicadores, pd.DataFrame) or not req.issubset(indicadores.columns) or len(indicadores) < 3:
        return False
    try:
        candle = indicadores.iloc[-1]
        corpo = abs(candle["Close"] - candle["Open"])
        pavio_superior = candle["High"] - max(candle["Close"], candle["Open"])

        cond_pavio = pavio_superior > corpo * 1.2
        cond_mm = candle["Close"] < indicadores["SMA20"].iloc[-1]
        cond_volume = (
            indicadores["Volume"].iloc[-1]
            < indicadores["Volume"].iloc[-2]
            < indicadores["Volume"].iloc[-3]
        )
        return bool(cond_pavio and cond_mm and cond_volume)
    except Exception:
        return False


def validar_reversao_alta(indicadores: pd.DataFrame) -> bool:
    """
    Reversão de baixa -> alta:
    - martelo (pavio inferior longo)
    - fechamento acima da SMA20
    - volume crescente nos últimos 3 candles
    """
    req = {"Open", "High", "Low", "Close", "SMA20", "Volume"}
    if not isinstance(indicadores, pd.DataFrame) or not req.issubset(indicadores.columns) or len(indicadores) < 3:
        return False
    try:
        candle = indicadores.iloc[-1]
        corpo = abs(candle["Close"] - candle["Open"])
        pavio_inferior = min(candle["Close"], candle["Open"]) - candle["Low"]

        cond_martelo = pavio_inferior > corpo * 1.5
        cond_mm = candle["Close"] > indicadores["SMA20"].iloc[-1]
        cond_volume = (
            indicadores["Volume"].iloc[-1]
            > indicadores["Volume"].iloc[-2]
            > indicadores["Volume"].iloc[-3]
        )
        return bool(cond_martelo and cond_mm and cond_volume)
    except Exception:
        return False


def calcular_grau_confianca(
    tendencia_combinada: str,
    microtendencia: str,
    reversao_confirmada: bool,
) -> str:
    """
    Define o grau de confiança com base em:
    - Convergência/divergência da tendência
    - Microtendência projetada
    - Reversão técnica confirmada
    """
    tendencia = (tendencia_combinada or "").lower()
    micro = (microtendencia or "").lower()

    if not reversao_confirmada:
        grau = "Baixa"
        print(">> grau_confiança:", grau)
        return grau

    if "convergente de baixa" in tendencia or "convergente de alta" in tendencia:
        if any(p in micro for p in ["+0", "+", "-0", "-"]):
            grau = "Alta"
        elif "estabilidade" in micro or "estável" in micro:
            grau = "Média"
        else:
            grau = "Média"
        print(">> grau_confiança:", grau)
        return grau

    if "divergente" in tendencia:
        grau = "Baixa"
        print(">> grau_confiança:", grau)
        return grau

    grau = "Média"
    print(">> grau_confiança:", grau)
    return grau


# =============================
# 4) Features para ML (GBM)
# =============================

def features_para_ml(df: pd.DataFrame):
    """
    Retorna X, y para classificação direcional.
    y = 1 se Close(t+1) > Close(t), do contrário 0.
    """
    d = df.copy()

    cols_req = [
        "Close", "SMA20", "SMA50", "UpperBand", "LowerBand",
        "RSI", "MACD", "MACD_Signal", "Volume_Medio"
    ]
    for c in cols_req:
        if c not in d.columns:
            d[c] = np.nan

    d["ret"] = d["Close"].pct_change().fillna(0.0)
    d["pctb"] = (d["Close"] - d["LowerBand"]) / (d["UpperBand"] - d["LowerBand"] + 1e-9)
    d["slope20"] = d["SMA20"].diff()
    d["slope50"] = d["SMA50"].diff()

    d = d.dropna().copy()
    if len(d) < 3:
        # muito pouco dado após limpeza
        return np.empty((0, 8)), np.empty((0,))

    X = d[["RSI", "MACD", "MACD_Signal", "pctb", "slope20", "slope50", "Volume_Medio", "ret"]].values
    y = (d["Close"].shift(-1) > d["Close"]).astype(int).iloc[:-1].values
    X = X[:-1]
    return X, y