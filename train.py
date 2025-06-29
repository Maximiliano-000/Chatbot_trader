import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from model import criar_modelo

dados = yf.download("PETR4.SA", period='1y')['Close'].dropna().values.reshape(-1, 1)
scaler = MinMaxScaler()
dados_norm = scaler.fit_transform(dados)

janela = 20
X, y = [], []
for i in range(janela, len(dados_norm)):
    X.append(dados_norm[i-janela:i, 0])
    y.append(dados_norm[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

modelo = criar_modelo((X.shape[1], 1))
modelo.fit(X, y, epochs=50, batch_size=32)

modelo.save('models/modelo_lstm.keras')