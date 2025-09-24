from __future__ import annotations
import os
import math
import pandas as pd

DEFAULT = {
    "15min": {"prophet": 0.05, "lstm": 0.65, "indicadores": 0.30},
    "30min": {"prophet": 0.05, "lstm": 0.65, "indicadores": 0.30},
    "1h":    {"prophet": 0.10, "lstm": 0.60, "indicadores": 0.30},
    "1d":    {"prophet": 0.25, "lstm": 0.45, "indicadores": 0.30},
    "1sem":  {"prophet": 0.40, "lstm": 0.30, "indicadores": 0.30},
    "1mes":  {"prophet": 0.50, "lstm": 0.20, "indicadores": 0.30},
}

def _normalize(wp, wl, wi):
    w = max(wp, 0) + max(wl, 0) + max(wi, 0)
    if w == 0: return (0.33, 0.33, 0.34)
    return (max(wp,0)/w, max(wl,0)/w, max(wi,0)/w)

def obter_pesos(intervalo: str, historico="avaliacoes/score_completo.csv", usar_indicadores_min=0.20):
    """
    Calcula pesos por intervalo com base no histórico de RMSE recente.
    """
    if not os.path.exists(historico):
        return DEFAULT.get(intervalo, {"prophet":.1,"lstm":.6,"indicadores":.3})
    try:
        df = pd.read_csv(historico)
        df = df.dropna(subset=["RMSE_Prophet","RMSE_LSTM"])
        if "intervalo" in df.columns:
            df = df[df["intervalo"].astype(str).str.lower()==intervalo.lower()]
        df = df.tail(200)  # janela recente
        if df.empty: 
            return DEFAULT.get(intervalo, {"prophet":.1,"lstm":.6,"indicadores":.3})
        # inverso do erro -> maior peso para menor RMSE
        inv_p = 1.0 / (df["RMSE_Prophet"].mean() + 1e-9)
        inv_l = 1.0 / (df["RMSE_LSTM"].mean() + 1e-9)
        # reserva mínima para “indicadores”
        wi_min = usar_indicadores_min
        wp, wl, wi = _normalize(inv_p, inv_l, wi_min)
        return {"prophet": round(wp,2), "lstm": round(wl,2), "indicadores": round(wi,2)}
    except Exception:
        return DEFAULT.get(intervalo, {"prophet":.1,"lstm":.6,"indicadores":.3})
