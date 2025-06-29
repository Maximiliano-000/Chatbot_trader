
import os
import importlib
import sys
import platform

def checar_pacote(nome):
    try:
        importlib.import_module(nome)
        return True
    except ImportError:
        return False

def verificar_pacotes_essenciais():
    print("🔍 Verificando pacotes essenciais:")
    pacotes = ["pandas", "prophet", "binance", "tensorflow"]
    for pacote in pacotes:
        status = "✅" if checar_pacote(pacote) else "❌"
        print(f"{status} {pacote}")

def verificar_estrutura():
    print("\n📂 Verificando estrutura de diretórios e arquivos:")
    pastas_esperadas = ["dados", "avaliacoes", "previsoes_prophet"]
    for pasta in pastas_esperadas:
        if os.path.isdir(pasta):
            print(f"✅ Pasta encontrada: {pasta}/")
        else:
            print(f"⚠️ Pasta ausente: {pasta}/")

    arquivos_recomendados = [
        "main.py",
        "prophet_forecaster.py",
        "avaliador_completo.py",
        "coletar_precos_binance.py"
    ]
    for arq in arquivos_recomendados:
        if os.path.isfile(arq):
            print(f"✅ Arquivo presente: {arq}")
        else:
            print(f"❌ Arquivo ausente: {arq}")

def verificar_ambiente():
    print("\n💻 Ambiente Python:")
    print(f"Versão do Python: {platform.python_version()}")
    print(f"Ambiente virtual ativo: {'analiz_arm' if 'analiz_arm' in sys.prefix else sys.prefix}")

def checar_arquivo_csv(nome_arquivo):
    print(f"\n📄 Verificando arquivo de dados: {nome_arquivo}")
    if not os.path.isfile(nome_arquivo):
        print(f"❌ Arquivo não encontrado: {nome_arquivo}")
        sugestoes = os.listdir("dados") if os.path.isdir("dados") else []
        print("📂 Arquivos disponíveis em /dados:")
        for s in sugestoes:
            print("  -", s)
        return
    else:
        print(f"✅ Arquivo encontrado: {nome_arquivo}")
        try:
            import pandas as pd
            df = pd.read_csv(nome_arquivo)
            print(f"🧾 Linhas carregadas: {len(df)}")
            print(f"📅 Datas (início → fim): {df['ds'].min()} → {df['ds'].max()}")
        except Exception as e:
            print("⚠️ Erro ao carregar CSV:", e)

if __name__ == "__main__":
    verificar_ambiente()
    verificar_pacotes_essenciais()
    verificar_estrutura()
    checar_arquivo_csv("dados/precos_pendleUSD.csv")
