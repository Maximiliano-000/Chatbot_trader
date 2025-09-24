# utils/moeda.py
from __future__ import annotations
import re
from typing import Literal, Callable

Mercado = Literal["CRYPTO", "B3", "OUTRO"]

def detectar_mercado(ticker: str | None) -> Mercado:
    """
    CRYPTO: exemplos de entrada: 'BTC-USD', 'ETH-USD', 'pendle-USD', 'PENDLEUSDT'
    B3:     'ABEV3', 'VALE3', 'BOVA11', 'PETR4', 'PETR4.SA'
    OUTRO:  fallback
    """
    t = (ticker or "").upper().strip()
    if not t:
        return "OUTRO"

    # heurísticas simples e robustas para cripto
    if any(s in t for s in ("-USD", "/USD", "USDT")) or "-" in t or t.endswith("USD"):
        return "CRYPTO"

    # heurísticas para B3 (4 letras + 1-2 dígitos) ou .SA
    if t.endswith(".SA") or re.fullmatch(r"[A-Z]{4}\d{1,2}", t):
        return "B3"

    return "OUTRO"

def moeda_por_mercado(mercado: Mercado) -> str:
    return "USD" if mercado == "CRYPTO" else "BRL"

def simbolo_moeda(sigla: str) -> str:
    return "US$" if sigla == "USD" else "R$"

def fmt_moeda(valor: float | int | None, sigla: str) -> str:
    """
    Formata visualmente SEM afetar cálculos internos.
    - USD:  US$ 12,345.67
    - BRL:  R$ 12.345,67
    """
    if valor is None:
        return "-"
    try:
        v = float(valor)
    except Exception:
        return "-"
    if sigla == "USD":
        return f"US$ {v:,.2f}"  # separador padrão en_US
    # BRL (pt-BR): troca separadores
    s = f"{v:,.2f}"                # 12,345.67
    s_pt = s.replace(",", "X").replace(".", ",").replace("X", ".")  # 12.345,67
    return f"R$ {s_pt}"

def make_fmt(sigla: str) -> Callable[[float | int | None], str]:
    """Conveniente para injetar no escopo local: fmt = make_fmt('BRL'|'USD')."""
    return lambda v: fmt_moeda(v, sigla)