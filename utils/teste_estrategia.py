import csv
import os
from datetime import datetime
from utils.indicadores import calcular_estrategia_longa, calcular_estrategia_short

# Simula diferentes formas de ATR que poderiam quebrar o sistema
testes_atr = [
    2.0,
    None,
    {"valor": 2.5},
    {"erro": "falha"},
    "2.5",
    0
]

preco_atual = 100.00
ticker = "SOL-USD"

resultados = []

print("\n=== Teste Estratégia Longa ===")
sucesso_longa = 0
for i, atr in enumerate(testes_atr, 1):
    try:
        resultado = calcular_estrategia_longa(preco_atual, atr=atr, ticker=ticker)
        assert isinstance(resultado, dict), "Resultado deve ser um dicionário"
        assert all(k in resultado for k in ["tp1", "tp2", "tp3", "sl"]), "Faltam chaves esperadas"
        assert all(isinstance(resultado[k], float) for k in ["tp1", "tp2", "tp3", "sl"]), "Todos os valores devem ser float"
        sucesso_longa += 1
        print(f"\n✅ Teste {i} - ATR: {atr}\n", resultado)
        resultados.append(["Long", atr, "OK", resultado])
    except Exception as e:
        print(f"\n❌ Teste {i} - ATR: {atr}\n[ERRO]: {e}")
        resultados.append(["Long", atr, "ERRO", str(e)])

print("\n=== Teste Estratégia Short ===")
sucesso_short = 0
for i, atr in enumerate(testes_atr, 1):
    try:
        resultado = calcular_estrategia_short(preco_atual, atr=atr, ticker=ticker)
        assert isinstance(resultado, dict), "Resultado deve ser um dicionário"
        assert all(k in resultado for k in ["tp1", "tp2", "tp3", "sl"]), "Faltam chaves esperadas"
        assert all(isinstance(resultado[k], float) for k in ["tp1", "tp2", "tp3", "sl"]), "Todos os valores devem ser float"
        sucesso_short += 1
        print(f"\n✅ Teste {i} - ATR: {atr}\n", resultado)
        resultados.append(["Short", atr, "OK", resultado])
    except Exception as e:
        print(f"\n❌ Teste {i} - ATR: {atr}\n[ERRO]: {e}")
        resultados.append(["Short", atr, "ERRO", str(e)])

# Exporta para CSV com timestamp na pasta relatorios/
agora = datetime.now().strftime("%Y-%m-%d_%H%M")
relatorio_dir = "relatorios"
os.makedirs(relatorio_dir, exist_ok=True)
nome_arquivo = os.path.join(relatorio_dir, f"relatorio_teste_{agora}.csv")

with open(nome_arquivo, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Tipo", "ATR", "Status", "Resultado"])
    writer.writerows(resultados)

print("\n=== Resumo Final ===")
print(f"✅ Longa: {sucesso_longa}/{len(testes_atr)} testes passaram")
print(f"✅ Short: {sucesso_short}/{len(testes_atr)} testes passaram")
print(f"📁 Arquivo gerado: {nome_arquivo}")

# Visual com barras
barra_ok = "█"
barra_erro = "░"

total = len(testes_atr)
barra_longa = barra_ok * sucesso_longa + barra_erro * (total - sucesso_longa)
barra_short = barra_ok * sucesso_short + barra_erro * (total - sucesso_short)

print(f"\nVisual Longa : [{barra_longa}] ({sucesso_longa}/{total})")
print(f"Visual Short : [{barra_short}] ({sucesso_short}/{total})")