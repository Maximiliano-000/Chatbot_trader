def gerar_explicacao_estrategia(tipo, contexto, media_ponderada, fibonacci, microtendencia, tendencia_combinada):
    """
    Gera uma explicação estratégica refinada, adaptada ao tipo (short, long, neutro),
    levando em conta contexto técnico, microtendência e convergência dos modelos.
    """
    if tipo == "short":
        resistencia_fib = None
        for nivel, preco in fibonacci.items():
            try:
                if preco is not None and abs(media_ponderada - preco) <= 0.5 and float(nivel.replace("%", "")) < 61.8:
                    resistencia_fib = f"nível de Fibonacci de {nivel} (R$ {preco:.2f})"
                    break
            except:
                continue

        msg = "📉 **Justificativa Estratégica:** O ativo apresenta sinais de sobrecompra"
        if resistencia_fib:
            msg += f", com média ponderada próxima ao {resistencia_fib}"
        msg += ". "
        if "baixa" in microtendencia.lower():
            msg += "A microtendência reforça o enfraquecimento no curto prazo. "
        else:
            msg += "A microtendência sugere estabilidade ou leve reversão. "
        msg += f"A convergência entre os modelos indica uma **{tendencia_combinada.lower()}** com possível perda de força compradora. "
        msg += "Este cenário favorece uma venda estratégica com pontos de realização técnica e proteção via stop, respeitando a resistência observada."

        return msg

    elif tipo == "long":
        suporte_fib = None
        for nivel, preco in fibonacci.items():
            try:
                if preco is not None and abs(media_ponderada - preco) <= 0.5 and float(nivel.replace("%", "")) > 38.2:
                    suporte_fib = f"nível de Fibonacci de {nivel} (R$ {preco:.2f})"
                    break
            except:
                continue

        msg = "📈 **Justificativa Estratégica:** O ativo encontra-se em possível zona de reversão"
        if suporte_fib:
            msg += f", com média ponderada próxima ao {suporte_fib}"
        msg += ". "
        if "alta" in microtendencia.lower():
            msg += "A microtendência projeta reação positiva no curto prazo. "
        else:
            msg += "A microtendência ainda sugere instabilidade leve. "
        msg += f"Com base na análise combinada, identifica-se uma **{tendencia_combinada.lower()}** com viés de recuperação. "
        msg += "Este cenário sustenta uma estratégia de compra com alvos técnicos progressivos e stop loss abaixo do suporte observado."

        return msg

    else:
        return "🔍 A situação técnica atual não oferece sinais claros de entrada. Acompanhar a formação de novos gatilhos e confirmar direção nos próximos candles."


def gerar_conclusao_dinamica(tendencia, rsi, preco_atual, sma20):
    """
    Gera uma conclusão final coerente com os dados técnicos e a tendência observada.
    """
    if rsi > 70:
        return (
            "📉 **Cenário técnico sugere cautela:** O ativo apresenta forte sobrecompra e pode entrar em fase de correção. "
            "Usuários com posição comprada devem avaliar realização parcial dos lucros ou ajuste de proteção."
        )
    elif rsi > 60:
        return (
            "📉 **Sinal de sobrecompra moderada:** Embora ainda haja força compradora, o ativo se aproxima de zonas de exaustão. "
            "Recomenda-se atenção redobrada a sinais de reversão e resistência técnica."
        )
    elif rsi < 30:
        return (
            "📈 **Sinal de sobrevenda acentuada:** O ativo pode apresentar reação técnica nos próximos candles. "
            "Compradores mais arrojados podem observar possíveis gatilhos de entrada com cautela."
        )
    elif rsi < 40:
        return (
            "📈 **Possível reversão técnica:** O ativo mostra sinais de enfraquecimento na pressão vendedora. "
            "O cruzamento do preço acima da média pode indicar início de recuperação."
        )
    elif abs(preco_atual - sma20) < preco_atual * 0.01:
        return (
            "🔁 **Cenário neutro:** O preço está próximo da média móvel, sem sinais claros de rompimento. "
            "Acompanhar movimentações com volume e gatilhos adicionais."
        )
    else:
        return (
            "🔍 **Cenário técnico em consolidação:** Não há sinais extremos no momento. "
            "Recomenda-se acompanhar a evolução dos indicadores para confirmar direção."
        )