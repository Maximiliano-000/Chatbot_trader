

def preparar_dados_prophet(dados):
    dados = dados.copy().reset_index()
    dados.rename(columns={'Date': 'ds', 'Datetime': 'ds', 'datetime': 'ds', 'Close': 'y', 'close': 'y'}, inplace=True)

    if 'ds' not in dados.columns or 'y' not in dados.columns:
        raise ValueError("Colunas obrigatórias ('ds', 'y') não encontradas nos dados.")

    # 🔒 Garantir Volume mesmo que não exista
    if 'Volume' not in dados.columns or dados['Volume'].dropna().empty:
        dados['Volume'] = 0.0  # Preenchimento neutro
    else:
        dados['Volume'] = dados['Volume'].fillna(method='ffill').fillna(method='bfill')

    dados = dados[['ds', 'y', 'Volume']].dropna()

    return dados
