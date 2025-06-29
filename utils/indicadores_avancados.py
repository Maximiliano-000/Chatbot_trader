import pandas as pd
import numpy as np

def calcular_adx(df, periodo=14):
    df = df.copy()
    if len(df) < periodo:
        print(f"[ERRO ADX] Dados insuficientes ({len(df)} candles)")
        return 0.0

    df['TR'] = np.maximum(df['High'] - df['Low'], 
                 np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                            abs(df['Low'] - df['Close'].shift(1))))
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 
                         np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 
                         np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)

    tr_smooth = df['TR'].rolling(window=periodo).sum()
    plus_dm_smooth = df['+DM'].rolling(window=periodo).sum()
    minus_dm_smooth = df['-DM'].rolling(window=periodo).sum()

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=periodo).mean()

    return round(adx.iloc[-1], 2)

def calcular_cci(df, periodo=20):
    df = df.copy()
    if len(df) < periodo:
        print(f"[ERRO CCI] Dados insuficientes ({len(df)} candles)")
        return 0.0

    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(window=periodo).mean()
    md = tp.rolling(window=periodo).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci = (tp - ma) / (0.015 * md)
    return round(cci.iloc[-1], 2)

def calcular_vwap(df, silenciar=False):
    df = df.copy()
    if df.empty or 'Volume' not in df.columns or df['Volume'].sum() == 0:
        if not silenciar:
            print(f"[ERRO VWAP] Dados insuficientes ou volume zero")
        return 0.0

    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['TP'] * df['Volume']
    vwap = df['VP'].cumsum() / df['Volume'].cumsum()
    return round(vwap.iloc[-1], 2)

def calcular_atr(df, periodo=14):
    df = df.copy()
    if len(df) < periodo:
        print(f"[ERRO ATR] Dados insuficientes ({len(df)} candles)")
        return 0.0

    df['TR'] = np.maximum(df['High'] - df['Low'],
                 np.maximum(abs(df['High'] - df['Close'].shift(1)),
                            abs(df['Low'] - df['Close'].shift(1))))
    atr = df['TR'].rolling(window=periodo).mean()
    return round(atr.iloc[-1], 2)