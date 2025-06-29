# utils/cv_utils.py

def gerar_janelas_cv(n: int, freq: str) -> tuple[str, str, str]:
    """
    Gera init_str, period_str, horizon_str com base no número de registros `n`
    e na frequência dos dados ("minutes", "hours" ou "days").
    """
    unidade = freq
    f = {
        "minutes": 1,
        "hours": 60,
        "days": 1440
    }[unidade]

    initial = max(3, int(n * 0.6))
    horizon = max(1, int(n * 0.1))
    period  = max(1, int((n - initial - horizon) / 3))

    init_str    = f"{initial * f} {unidade}"
    period_str  = f"{period * f} {unidade}"
    horizon_str = f"{horizon * f} {unidade}"

    return init_str, period_str, horizon_str