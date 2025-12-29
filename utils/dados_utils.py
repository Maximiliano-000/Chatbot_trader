import pandas as pd
import numpy as np

def preparar_dados_prophet(dados: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara dados no formato exigido pelo Prophet:
      - colunas: ds (datetime naive), y (float), Volume (float)
      - tipos coerentes
      - sem NaN em ds/y
      - volume com fallback para 0.0 quando ausente
      - ordenado e sem duplicatas em ds
    """
    df = dados.copy().reset_index()

    # Normaliza nomes comuns de colunas
    rename_map = {
        'Date': 'ds', 'Datetime': 'ds', 'datetime': 'ds',
        'Close': 'y', 'close': 'y'
    }
    df.rename(columns=rename_map, inplace=True)

    # Se ainda não existir 'ds' e seu índice original era temporal, tenta aproveitá-lo
    if 'ds' not in df.columns and 'index' in df.columns:
        df.rename(columns={'index': 'ds'}, inplace=True)

    # Checagem obrigatória
    if 'ds' not in df.columns or 'y' not in df.columns:
        raise ValueError("Colunas obrigatórias ('ds', 'y') não encontradas nos dados.")

    # Tipos corretos
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce', utc=True).dt.tz_localize(None)
    df['y']  = pd.to_numeric(df['y'], errors='coerce')

    # Volume com fallback e sem FutureWarning
    if 'Volume' in df.columns:
        vol = pd.to_numeric(df['Volume'], errors='coerce')
        # forward/backward fill e, se ainda faltar, zera
        df['Volume'] = vol.ffill().bfill().fillna(0.0)
    else:
        # cria a coluna com zeros (neutro) se não existir
        df['Volume'] = 0.0

    # 🔧 refinamento: garante float64 sempre
    df['Volume'] = df['Volume'].astype('float64', copy=False)

    # Drop de linhas inválidas e ordenação
    df = df.dropna(subset=['ds', 'y']).sort_values('ds')

    # Remove duplicatas de timestamp (mantém a última observação)
    df = df.drop_duplicates(subset='ds', keep='last')

    # Apenas as colunas necessárias (ordem esperada)
    df = df[['ds', 'y', 'Volume']].reset_index(drop=True)

    return df

