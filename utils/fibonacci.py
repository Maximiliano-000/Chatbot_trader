def interpretar_fibonacci_decisorio(
    media_ponderada: float,
    fibonacci: dict,
    fmt_preco,
    tol_pct: float = 0.0075,
    excluir_extremos: bool = True
) -> dict:
    """
    Interpreta Fibonacci de forma decisória (não descritiva).

    Retorna um dicionário com:
    - existe_confluencia (bool)
    - nivel (str ou None)
    - preco (str formatado)
    - tipo ("suporte" | "resistencia" | None)
    - desvio_pct (float ou None)
    - comentario (str pronto para o relatório)

    ❗ Deve ser a ÚNICA função que interpreta Fibonacci no projeto.
    """

    if not fibonacci or media_ponderada is None:
        return {
            "existe_confluencia": False,
            "nivel": None,
            "preco": None,
            "tipo": None,
            "desvio_pct": None,
            "comentario": "Sem confluência relevante com Fibonacci no momento."
        }

    candidatos = []
    for nivel, preco in fibonacci.items():
        try:
            if preco is None:
                continue

            nivel_f = float(nivel.replace("%", ""))
            if excluir_extremos and nivel_f in (0.0, 100.0):
                continue

            preco_f = float(preco)
            if preco_f <= 0:
                continue

            diff_pct = abs(media_ponderada - preco_f) / preco_f
            candidatos.append((nivel, preco_f, diff_pct))
        except Exception:
            continue

    if not candidatos:
        return {
            "existe_confluencia": False,
            "nivel": None,
            "preco": None,
            "tipo": None,
            "desvio_pct": None,
            "comentario": "Sem confluência relevante com Fibonacci no momento."
        }

    # Seleciona o nível mais próximo
    nivel, preco_f, diff_pct = min(candidatos, key=lambda x: x[2])

    # Fora da tolerância → ignora
    if diff_pct > tol_pct:
        return {
            "existe_confluencia": False,
            "nivel": None,
            "preco": None,
            "tipo": None,
            "desvio_pct": None,
            "comentario": "Sem confluência relevante com Fibonacci no momento."
        }

    # Classificação: suporte ou resistência
    tipo_nivel = "suporte" if preco_f < media_ponderada else "resistencia"

    comentario = (
        f"Média ponderada próxima do Fibonacci {nivel} "
        f"({fmt_preco(preco_f)}; desvio {diff_pct*100:.2f}%), "
        f"atuando como possível **{tipo_nivel} técnico**. "
        "Priorize confirmação por preço e volume."
    )

    return {
        "existe_confluencia": True,
        "nivel": nivel,
        "preco": fmt_preco(preco_f),
        "tipo": tipo_nivel,
        "desvio_pct": round(diff_pct * 100, 2),
        "comentario": comentario
    }
