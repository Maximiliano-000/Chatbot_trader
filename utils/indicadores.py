import numpy as np
import pandas as pd

# =============================
# 1. Calcular indicadores técnicos
# =============================
"""
    Calcula e retorna indicadores técnicos como médias móveis (SMA20, SMA50), 
    Bandas de Bollinger, RSI, MACD e Volume Médio, com base nos dados fornecidos.
"""

def calcular_indicadores(dados, intervalo='1d'):
    df = dados.copy()
    df.index = pd.to_datetime(df.index)

    if 'Close' not in df.columns:
        return pd.DataFrame()

    min_candles_por_intervalo = {
        '15min': 20,
        '30min': 25,
        '45min': 30,
        '1h': 35,
        '2h': 30,
        '6h': 30,
        '1d': 52
    }

    min_candles = min_candles_por_intervalo.get(intervalo, 30)

    if len(df) < min_candles:
        print(f"[ERRO] Dados insuficientes ({len(df)} candles). Mínimo exigido: {min_candles}.")
        return pd.DataFrame()

    if df['Close'].isnull().all():
        print("[ERRO] Todos valores de 'Close' são NaN.")
        return pd.DataFrame()

    # Indicadores
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()

    df['STD20'] = df['Close'].rolling(window=20).std()
    df['UpperBand'] = df['SMA20'] + (2 * df['STD20'])
    df['LowerBand'] = df['SMA20'] - (2 * df['STD20'])

    delta = df['Close'].diff()
    ganho = delta.clip(lower=0).rolling(window=14).mean()
    perda = -delta.clip(upper=0).rolling(window=14).mean()
    rs = ganho / (perda + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    if 'Volume' in df.columns:
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    
        # Substitua volumes zerados por NaN antes da média
        df.loc[df['Volume'] == 0, 'Volume'] = np.nan
    
        if df['Volume'].notna().any():
            df['Volume_Medio'] = df['Volume'].rolling(window=21, min_periods=1).mean()
        else:
            print("[AVISO] Não há valores válidos para cálculo de Volume Médio.")
            df['Volume_Medio'] = np.nan
    else:
        print("[AVISO] Coluna 'Volume' não encontrada.")
        df['Volume_Medio'] = np.nan

    colunas_float = ['Close', 'SMA20', 'SMA50', 'UpperBand', 'LowerBand', 'RSI', 'MACD', 'MACD_Signal', 'Volume_Medio']
    df[colunas_float] = df[colunas_float].apply(pd.to_numeric, errors='coerce')

    # Importante: remova apenas se Close estiver vazio, mantenha indicadores
    df = df.dropna(subset=['Close'])

    if df.empty:
        print("[ERRO] DataFrame vazio após remover 'Close' nulos.")
        return pd.DataFrame()

    # Aceite NaNs em indicadores e faça preenchimento suave/interpolação
    df[colunas_float] = df[colunas_float].interpolate(method='linear', limit_direction='both').ffill().bfill()

    return df

def calcular_fibonacci(preco_min, preco_max):
    """
    Retorna os principais níveis de retração de Fibonacci.
    """
    diff = preco_max - preco_min
    return {
        "0.0%": round(preco_max, 2),
        "23.6%": round(preco_max - 0.236 * diff, 2),
        "38.2%": round(preco_max - 0.382 * diff, 2),
        "50.0%": round(preco_max - 0.5 * diff, 2),
        "61.8%": round(preco_max - 0.618 * diff, 2),
        "100.0%": round(preco_min, 2)
    }

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
    sobrevenda=False
):
    def avaliar_condicoes_estrategia_compra(
        rsi,
        candle_reversao,
        volume_crescente,
        suporte_fibo,
        cruzamento_macd,
        divergencia_rsi,
        sobrevenda
    ):
        sinais_forca = any([
            candle_reversao,
            volume_crescente,
            suporte_fibo,
            cruzamento_macd,
            divergencia_rsi
        ])

        if sobrevenda and suporte_fibo and sinais_forca:
            return {
                "tipo": "Long",
                "mensagem": "Compra sugerida com base em sobrevenda, suporte técnico e sinais confirmados de força compradora."
            }
        elif sobrevenda and suporte_fibo:
            return {
                "tipo": "Observacao",
                "mensagem": "Zona de atenção técnica: sobrevenda e suporte identificados, mas sem sinais claros de força. Aguardar confirmação."
            }
        else:
            return {
                "tipo": "Neutra",
                "mensagem": "Sem evidências suficientes para sugerir compra no momento."
            }

    avaliacao = avaliar_condicoes_estrategia_compra(
        rsi,
        candle_reversao,
        volume_crescente,
        suporte_fibo,
        cruzamento_macd,
        divergencia_rsi,
        sobrevenda
    )

    try:
        atr = float(atr) if isinstance(atr, (int, float, str)) else atr.get("valor", 0)
    except (TypeError, ValueError, AttributeError):
        atr = 0.0

    entrada = round(preco_atual, 2)
    from .multiplicador import obter_multiplicador_atr
    multiplicador = obter_multiplicador_atr(ticker) if ticker else 1.5

    alvos_modelos = []
    if previsao_lstm and len(previsao_lstm) > 0:
        alvos_modelos.append(max(previsao_lstm))
    if previsao_prophet is not None and 'yhat_upper' in previsao_prophet.columns:
        alvos_modelos.append(previsao_prophet['yhat_upper'].max())

    alvo_max_modelos = max(alvos_modelos) if alvos_modelos else preco_atual * 1.03
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
        "sl": sl
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
    resistencia_fibo=False
):
    def avaliar_condicoes_estrategia_venda(
        rsi,
        candle_reversao,
        volume_decrescente,
        rejeicao_resistencia,
        cruzamento_macd,
        divergencia_rsi,
        sobrecompra,
        resistencia_fibo
    ):
        sinais_fraqueza = any([
            candle_reversao,
            volume_decrescente,
            rejeicao_resistencia,
            cruzamento_macd,
            divergencia_rsi
        ])

        if sobrecompra and resistencia_fibo and sinais_fraqueza:
            return {
                "tipo": "Short",
                "mensagem": "Venda sugerida com base em sobrecompra, resistência e sinais confirmados de fraqueza (reversão ou exaustão)."
            }
        elif sobrecompra and resistencia_fibo:
            return {
                "tipo": "Observacao",
                "mensagem": "Zona de atenção técnica: sobrecompra e resistência detectadas, mas sem sinais claros de reversão. Aguardar confirmação."
            }
        else:
            return {
                "tipo": "Neutra",
                "mensagem": "Sem evidências suficientes para sugerir venda no momento."
            }

    avaliacao = avaliar_condicoes_estrategia_venda(
        rsi,
        candle_reversao,
        volume_decrescente,
        rejeicao_resistencia,
        cruzamento_macd,
        divergencia_rsi,
        sobrecompra,
        resistencia_fibo
    )

    try:
        atr = float(atr) if isinstance(atr, (int, float, str)) else atr.get("valor", 0)
    except (TypeError, ValueError, AttributeError):
        atr = 0.0

    entrada = round(preco_atual, 2)
    from .multiplicador import obter_multiplicador_atr
    multiplicador = obter_multiplicador_atr(ticker) if ticker else 1.5

    alvos_modelos = []
    if previsao_lstm and len(previsao_lstm) > 0:
        alvos_modelos.append(min(previsao_lstm))
    if previsao_prophet is not None and 'yhat_lower' in previsao_prophet.columns:
        alvos_modelos.append(previsao_prophet['yhat_lower'].min())

    alvo_min_modelos = min(alvos_modelos) if alvos_modelos else preco_atual * 0.97
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
        "sl": sl
    }

def gerar_microtendencia(preco_atual: float, previsoes_lstm: list) -> str:
    """
    Gera uma frase interpretativa com base nas próximas previsões LSTM
    e no preço atual. Indica tendência leve, estabilidade ou projeção.
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

    # 🔍 Diagnóstico opcional
    # print(f"[Micro] direção: {direcao}, delta: {delta}, média: {media_prevista:.3f}, atual: {preco_atual:.3f}")

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

def validar_reversao_baixa(indicadores) -> bool:
    '''
    Confirma reversão de alta para queda com base em:
    - candle de rejeição superior (pavio superior longo)
    - fechamento abaixo da média curta (SMA20)
    - volume decrescente nos últimos 3 candles
    '''
    try:
        candle = indicadores.iloc[-1]
        corpo = abs(candle['Close'] - candle['Open'])
        pavio_superior = candle['High'] - max(candle['Close'], candle['Open'])

        cond_pavio = pavio_superior > corpo * 1.2
        cond_mm = candle['Close'] < indicadores['SMA20'].iloc[-1]
        cond_volume = (
            len(indicadores) > 3 and
            indicadores['Volume'].iloc[-1] < indicadores['Volume'].iloc[-2] < indicadores['Volume'].iloc[-3]
        )

        return cond_pavio and cond_mm and cond_volume
    except Exception:
        return False

def validar_reversao_alta(indicadores) -> bool:
    '''
    Confirma reversão de queda para alta com base em:
    - candle martelo (pavio inferior longo)
    - fechamento acima da média curta (SMA20)
    - volume crescente nos últimos 3 candles
    '''
    try:
        candle = indicadores.iloc[-1]
        corpo = abs(candle['Close'] - candle['Open'])
        pavio_inferior = min(candle['Close'], candle['Open']) - candle['Low']

        cond_martelo = pavio_inferior > corpo * 1.5
        cond_mm = candle['Close'] > indicadores['SMA20'].iloc[-1]
        cond_volume = (
            len(indicadores) > 3 and
            indicadores['Volume'].iloc[-1] > indicadores['Volume'].iloc[-2] > indicadores['Volume'].iloc[-3]
        )

        return cond_martelo and cond_mm and cond_volume
    except Exception:
        return False

def calcular_grau_confianca(
    tendencia_combinada: str,
    microtendencia: str,
    reversao_confirmada: bool
) -> str:
    """
    Define o grau de confiança com base em:
    - Convergência da tendência
    - Microtendência projetada
    - Reversão técnica confirmada (via candle + volume)
    """
    tendencia = tendencia_combinada.lower()
    micro = microtendencia.lower()

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

def features_para_ml(df):
    """
    Retorna X, y (opcional) para classificação direcional.
    y = 1 se Close(t+1) > Close(t), do contrário 0.
    """
    d = df.copy()
    cols_req = ["Close","SMA20","SMA50","UpperBand","LowerBand","RSI","MACD","MACD_Signal","Volume_Medio"]
    for c in cols_req:
        if c not in d.columns:
            d[c] = np.nan
    d["ret"] = d["Close"].pct_change().fillna(0.0)
    d["pctb"] = (d["Close"] - d["LowerBand"]) / (d["UpperBand"] - d["LowerBand"] + 1e-9)
    d["slope20"] = d["SMA20"].diff()
    d["slope50"] = d["SMA50"].diff()
    d = d.dropna().copy()
    X = d[["RSI","MACD","MACD_Signal","pctb","slope20","slope50","Volume_Medio","ret"]].values
    y = (d["Close"].shift(-1) > d["Close"]).astype(int).iloc[:-1].values
    X = X[:-1]
    return X, y
