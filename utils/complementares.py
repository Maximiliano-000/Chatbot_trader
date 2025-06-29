import pandas as pd

def gerar_cenarios_alternativos(preco_atual):
    """
    Retorna um DataFrame com cenários alternativos (alta moderada e queda acelerada),
    incluindo gatilhos técnicos, alvos e comentários, com base no preço atual.
    """
    preco_atual = float(preco_atual)
    gatilho_alta = round(preco_atual * 1.015, 2)
    alvo_alta = round(preco_atual * 1.05, 2)
    gatilho_baixa = round(preco_atual * 0.985, 2)
    alvo_baixa = round(preco_atual * 0.94, 2)

    df = pd.DataFrame([
        {"Cenário": "Alta moderada", 
         "Gatilho": f"Rompimento de R$ {gatilho_alta:.2f}", 
         "Alvo": f"R$ {alvo_alta:.2f}", 
         "Comentário": "Valida reversão"},

        {"Cenário": "Queda acelerada", 
         "Gatilho": f"Perda de R$ {gatilho_baixa:.2f}", 
         "Alvo": f"R$ {alvo_baixa:.2f}", 
         "Comentário": "Pressiona suporte"}
    ])

    return df


def ticker_formatado(ticker: str) -> str:
    """
    Formata o ticker removendo sufixos (.SA ou -USD).
    """
    return ticker.replace(".SA", "").replace("-USD", "")