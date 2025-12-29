from __future__ import annotations

import math
from utils.moeda import make_fmt
from utils.fibonacci import interpretar_fibonacci_decisorio


def gerar_explicacao_estrategia(
    tipo,
    contexto,
    media_ponderada,
    fibonacci,
    microtendencia,
    tendencia_combinada,
    sigla_moeda="USD",
    tol_fibo_pct=0.0075,
    moeda=None,
    **kwargs
):
    """
    Texto de estratégia com semântica correta.
    - LSTM: curto prazo (timing)
    - Prophet: estrutura (contexto)
    - tendencia_combinada: regime multi-horizonte

    Observação:
    - `moeda` é aceito para compatibilidade com a rota.
    - `**kwargs` blinda contra futuras refatorações.
    """

    tipo = (tipo or "").strip().lower()
    fmt = make_fmt(sigla_moeda)

    # Fibonacci decisório (API única)
    fib_txt = ""
    try:
        fibo_info = interpretar_fibonacci_decisorio(
            media_ponderada=media_ponderada,
            fibonacci=fibonacci,
            fmt_preco=fmt,
            tol_pct=tol_fibo_pct
        )
        if fibo_info and bool(fibo_info.get("existe_confluencia")):
            comentario = (fibo_info.get("comentario") or "").strip()
            if comentario:
                # garante pontuação natural
                fib_txt = (" " + comentario) if comentario.startswith(("(", "—")) else (" " + comentario)
                if not fib_txt.endswith((".", "!", "?")):
                    fib_txt += "."
    except Exception:
        fib_txt = ""

    tc = (tendencia_combinada or "").strip().lower()
    if "diverg" in tc:
        regime = "multi-horizonte (curto prazo tentando se mover contra uma estrutura ainda indefinida)"
        regra = "Somente operar com gatilho confirmado e alvos moderados; evitar convicção."
    elif "converg" in tc:
        regime = "convergente (curto prazo e estrutura apontando na mesma direção)"
        regra = "Operações permitem mais continuidade, mantendo disciplina de risco."
    else:
        regime = "indefinido"
        regra = "Priorizar espera e gatilhos mais claros."

    mt = (microtendencia or "").strip().lower()
    ctx = (contexto or "").strip()
    ctx_txt = f" {ctx} " if ctx else " "

    if tipo == "short":
        msg = "📉 **Justificativa Estratégica (Short):** "
        msg += "A leitura sugere perda de fôlego comprador no curto prazo."
        msg += ctx_txt

        if "baixa" in mt:
            msg += "A microtendência reforça enfraquecimento/pressão vendedora. "
        else:
            msg += "A microtendência não confirma aceleração de baixa; exigir confirmação. "

        msg += f"**Regime:** {regime}.{fib_txt} "
        msg += f"**Regra de execução:** {regra} "
        msg += "Estratégia: buscar rejeição em resistência/retomada de baixa; stop sempre acima do nível de invalidação."
        return msg

    if tipo == "long":
        msg = "📈 **Justificativa Estratégica (Long):** "
        msg += "A leitura indica tentativa de recuperação no curto prazo."
        msg += ctx_txt

        if "alta" in mt:
            msg += "A microtendência favorece reação positiva. "
        else:
            msg += "A microtendência ainda é fraca; exigir gatilho e confirmação. "

        msg += f"**Regime:** {regime}.{fib_txt} "
        msg += f"**Regra de execução:** {regra} "
        msg += "Estratégia: operar apenas após confirmação (fechamento/reteste); stop abaixo da invalidação estrutural."
        return msg

    return (
        "🔍 **Sem operação:** não há gatilho claro no momento. "
        "Acompanhar extremos do range, volume e confirmação de direção nos próximos candles."
    )


def gerar_conclusao_dinamica(tendencia, rsi, preco_atual, sma20, moeda=None, **kwargs):
    """
    Conclusão final coerente e auditável:
    - Exibe RSI e contexto (tendência).
    - Trata None/NaN.
    - Diferencia sobrecompra em tendência forte vs mercado lateral.

    Observação:
    - `moeda` é aceito para compatibilidade com chamadas nomeadas.
    - `**kwargs` blinda contra futuros parâmetros adicionados na rota.
    """
    import math

    try:
        rsi_val = float(rsi) if rsi is not None else None
        if rsi_val is not None and (math.isnan(rsi_val) or math.isinf(rsi_val)):
            rsi_val = None
    except Exception:
        rsi_val = None

    try:
        preco = float(preco_atual)
        sma = float(sma20)
    except Exception:
        return "🔍 **Conclusão indisponível:** dados insuficientes para avaliar o cenário com segurança."

    tend = (tendencia or "").strip().lower()
    rsi_txt = f"RSI {rsi_val:.1f}" if rsi_val is not None else "RSI indisponível"

    pos_sma = "acima" if preco > sma else "abaixo" if preco < sma else "em cima"
    sma_ctx = f"preço {pos_sma} da SMA20"

    if rsi_val is not None and rsi_val > 70:
        if "alta" in tend:
            return (
                f"📉 **Cautela com tendência de alta ({rsi_txt}; {sma_ctx}):** mercado esticado, "
                "mas ainda pode sustentar continuação. "
                "Evitar entrada tardia; preferir parcial e proteção (trailing/stop técnico)."
            )
        return (
            f"📉 **Cautela ({rsi_txt}; {sma_ctx}):** sobrecompra forte e risco de correção. "
            "Reduzir exposição e aguardar novo gatilho."
        )

    if rsi_val is not None and rsi_val > 60:
        if "alta" in tend:
            return (
                f"📈 **Força compradora ainda presente ({rsi_txt}; {sma_ctx}):** risco de pullback. "
                "Entradas só com confirmação; ajustar proteção."
            )
        return (
            f"📉 **Sobrecompra moderada ({rsi_txt}; {sma_ctx}):** atenção a resistência e sinais de reversão."
        )

    if rsi_val is not None and rsi_val < 30:
        return (
            f"📈 **Possível reação técnica ({rsi_txt}; {sma_ctx}):** sobrevenda acentuada. "
            "Observar gatilhos; entradas agressivas exigem stop técnico e tamanho reduzido."
        )

    if rsi_val is not None and rsi_val < 40:
        if "baixa" in tend:
            return (
                f"📉 **Venda ainda dominante ({rsi_txt}; {sma_ctx}):** alívio possível, sem reversão confirmada. "
                "Evitar antecipação; aguardar quebra de estrutura."
            )
        return (
            f"📈 **Venda enfraquecendo ({rsi_txt}; {sma_ctx}):** pode iniciar recuperação com confirmação. "
            "Aguardar fechamento + reteste antes de aumentar exposição."
        )

    if abs(preco - sma) < preco * 0.01:
        return (
            f"🔁 **Cenário neutro ({rsi_txt}; {sma_ctx}):** sem rompimento confirmado. "
            "Operar apenas em extremos do range ou após confirmação com volume."
        )

    if "alta" in tend:
        return (
            f"📈 **Viés positivo ({rsi_txt}; {sma_ctx}):** sem extremos. "
            "Favorecer compras com confirmação; evitar perseguir preço."
        )

    if "baixa" in tend:
        return (
            f"📉 **Viés negativo ({rsi_txt}; {sma_ctx}):** sem extremos. "
            "Shorts apenas com confirmação; evitar vender em suporte sem gatilho."
        )

    return (
        f"🔍 **Consolidação ({rsi_txt}; {sma_ctx}):** sem direção clara. "
        "Aguardar gatilhos e confirmação estrutural."
    )