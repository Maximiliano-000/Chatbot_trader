def test_import_analyz():
    import analyz
    # garante que o pacote importa e expõe versão
    assert hasattr(analyz, "__version__")

def test_import_submodules():
    # submódulos existem e importam
    import analyz.backtest
    import analyz.strategy
    assert True
