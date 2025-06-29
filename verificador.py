import os
import tensorflow as tf
import yfinance as yf
import socket
from lstm_forecaster import CriptoForecaster

def verificar_tensorflow():
    try:
        print(f"✅ TensorFlow OK: versão {tf.__version__}")
        if tf.config.list_physical_devices('GPU'):
            print("⚡️ GPU detectada e disponível para uso.")
        else:
            print("⚠️ GPU não detectada — usando CPU.")
    except Exception as e:
        print(f"❌ Erro no TensorFlow: {e}")

def verificar_yfinance():
    try:
        df = yf.download("BTC-USD", period="5d")
        if df.empty:
            raise ValueError("Sem dados retornados.")
        print("✅ yfinance OK: dados de BTC-USD carregados.")
    except Exception as e:
        print(f"❌ Erro no yfinance: {e}")

def verificar_modelo():
    try:
        forecaster = CriptoForecaster("BTC-USD", epochs=1)
        forecaster.carregar_dados()
        forecaster.treinar()
        previsoes = forecaster.prever(dias=3)
        print(f"✅ Modelo LSTM OK. Previsões: {previsoes}")
    except Exception as e:
        print(f"❌ Erro no modelo LSTM: {e}")

def verificar_modelo_salvo():
    path = "modelos_lstm/BTC-USD_modelo.keras"
    if os.path.exists(path):
        print(f"✅ Arquivo de modelo salvo encontrado: {path}")
    else:
        print(f"⚠️ Modelo salvo ainda não existe: {path}")

def verificar_internet():
    try:
        socket.gethostbyname("www.google.com")
        print("✅ Conexão com a internet detectada.")
    except:
        print("❌ Sem conexão com a internet.")

if __name__ == "__main__":
    print("🔍 Verificação do ambiente AnaliZ\n")
    verificar_tensorflow()
    verificar_internet()
    verificar_yfinance()
    verificar_modelo()
    verificar_modelo_salvo()