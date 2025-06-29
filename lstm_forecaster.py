import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from logger_perda import LoggerDePerda
from utils.dados_com_fallback import obter_dados_com_fallback

class CriptoForecaster:
    def __init__(self, ticker, janela=60, epochs=100, modelo_path=None):
        self.ticker = ticker
        self.janela = janela
        self.epochs = epochs
        self.modelo_path = modelo_path or f"modelos_lstm/{ticker}_modelo.keras"
        self.scaler_path = f"modelos_lstm/{ticker}_scaler.pkl"
        self.modelo = None
        self.scaler = MinMaxScaler()
        self.dados_treinamento = None

    def modelo_existente(self):
        return os.path.exists(self.modelo_path) and os.path.exists(self.scaler_path)

    def carregar_dados(self, preferencia="auto"):
        try:
            df, fonte, intervalo_usado, msg = obter_dados_com_fallback(
                self.ticker, intervalo="1d", periodo="1y", outputsize=365, preferencia=preferencia
            )

            if df.empty or "Close" not in df.columns or df["Close"].dropna().empty:
                raise ValueError(f"❌ Sem dados válidos para {self.ticker}")

            close = df["Close"].dropna().values.reshape(-1, 1)
            if len(close) < self.janela + 1:
                raise ValueError(f"⚠️ Dados insuficientes para {self.ticker} (len={len(close)})")

            self.dados_treinamento = self.scaler.fit_transform(close).flatten()
            print(f"📥 Dados carregados via {fonte} – {len(close)} pontos (intervalo: {intervalo_usado})")

        except Exception as e:
            raise RuntimeError(f"Erro ao carregar dados para {self.ticker}: {e}")

    def treinar(self, callbacks=None, arquitetura="simples"):
        X, y = self._preparar_dados_para_treino(self.dados_treinamento)

        self.modelo = Sequential()

        if arquitetura == "empilhada":
            self.modelo.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
            self.modelo.add(LSTM(50))
        else:
            self.modelo.add(LSTM(64, return_sequences=False, input_shape=(X.shape[1], 1)))

        self.modelo.add(Dense(1))
        self.modelo.compile(optimizer="adam", loss="mse")

        if callbacks is None:
            es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
            log = LoggerDePerda()
            callbacks = [es, log]

        self.modelo.fit(X, y, epochs=self.epochs, batch_size=16, verbose=1, callbacks=callbacks)
        self.salvar_modelo()

    def salvar_modelo(self):
        os.makedirs(os.path.dirname(self.modelo_path), exist_ok=True)
        self.modelo.save(self.modelo_path)
        joblib.dump(self.scaler, self.scaler_path)

    def carregar_modelo_treinado(self):
        if os.path.exists(self.modelo_path):
            self.modelo = load_model(self.modelo_path)
            self.scaler = joblib.load(self.scaler_path)
            self.carregar_dados()
        else:
            raise FileNotFoundError(f"Modelo não encontrado para {self.ticker}")

    def prever(self, dias=5):
        if not self.modelo:
            raise RuntimeError("Modelo ainda não carregado.")

        ultimos_dados = self.dados_treinamento[-self.janela:].copy()
        previsoes = []

        for _ in range(dias):
            entrada = np.array(ultimos_dados).reshape(1, self.janela, 1)
            proximo = self.modelo.predict(entrada, verbose=0)[0][0]

            if len(previsoes) >= 1:
                anterior = previsoes[-1]
                variacao = (proximo - anterior) / anterior
                if variacao > 0.03:
                    proximo = anterior * 1.03
                elif variacao < -0.03:
                    proximo = anterior * 0.97

            previsoes.append(proximo)
            ultimos_dados = np.append(ultimos_dados[1:], proximo)

        previsoes = np.array(previsoes).reshape(-1, 1)
        previsoes_reais = self.scaler.inverse_transform(previsoes).flatten()
        return [{"valor": round(v, 2)} for v in previsoes_reais]

    def _preparar_dados_para_treino(self, dados):
        X, y = [], []
        for i in range(len(dados) - self.janela):
            X.append(dados[i:i + self.janela])
            y.append(dados[i + self.janela])
        X = np.array(X).reshape(-1, self.janela, 1)
        y = np.array(y).reshape(-1, 1)
        return X, y
