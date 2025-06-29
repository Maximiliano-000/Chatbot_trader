import websocket
import json
import sqlite3

# Conecta ao banco SQLite
conn = sqlite3.connect('intraday.db')
cursor = conn.cursor()

# Cria tabela para armazenar dados intradiários
cursor.execute('''
CREATE TABLE IF NOT EXISTS order_flow (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    ticker TEXT,
    price REAL,
    quantity REAL,
    side TEXT
)
''')

conn.commit()

# Função para salvar os dados
def salvar_fluxo_ordens(ticker, price, quantity, side):
    cursor.execute('''
    INSERT INTO order_flow (ticker, price, quantity, side)
    VALUES (?, ?, ?, ?)
    ''', (ticker, price, quantity, side))
    conn.commit()

# Processamento dos dados recebidos
def on_message(ws, message):
    data = json.loads(message)

    if 's' in data and 'p' in data and 'q' in data and 'm' in data:
        ticker = data['s']
        price = float(data['p'])
        quantity = float(data['q'])
        side = 'Compra' if not data['m'] else 'Venda'

        print(f"Ticker: {ticker}, Price: {price}, Quantity: {quantity}, Side: {side}")

        salvar_fluxo_ordens(ticker, price, quantity, side)
    else:
        print(f"Mensagem inesperada recebida: {data}")

def on_error(ws, error):
    print(f"Erro: {error}")

def on_close(ws):
    print("Conexão encerrada")

def on_open(ws):
    params = []
    tickers = ['pendleusdt']  # Lista de tickers a monitorar

    for ticker in tickers:
        params.append(f"{ticker}@aggTrade")

    subscription = {
        "method": "SUBSCRIBE",
        "params": params,
        "id": 1
    }

    ws.send(json.dumps(subscription))

# Inicializa WebSocket
ws = websocket.WebSocketApp(
    "wss://stream.binance.com:9443/ws",
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)

ws.on_open = on_open
ws.run_forever()