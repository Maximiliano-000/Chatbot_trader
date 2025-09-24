# utils/metrics.py
import numpy as np
import pandas as pd

def directional_accuracy(y_true, y_pred):
    """
    Acurácia direcional: compara o sinal do retorno,
    i.e., se a direção prevista bate com a direção real.
    Retorna NaN se os vetores não tiverem tamanho suficiente.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or len(y_pred) != len(y_true):
        return np.nan

    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))

    # Ajuste defensivo se por algum motivo os comprimentos divergirem
    m = min(len(true_dir), len(pred_dir))
    if m == 0:
        return np.nan

    return float((true_dir[:m] == pred_dir[:m]).mean())

def brier_score(probs, outcomes):
    """
    Brier Score para probabilidade de alta (0..1) vs desfecho binário (0/1).
    Retorna NaN se tamanhos forem inválidos.
    """
    p = np.asarray(probs, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    if len(p) != len(o) or len(p) == 0:
        return np.nan
    return float(np.mean((p - o) ** 2))