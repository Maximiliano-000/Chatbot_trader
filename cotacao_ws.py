import websocket
import json

def on_message(ws, message):
    dados = json.loads(message)
    preco = float(dados['p'])
    print(f"[⏱️] Último preço ao vivo: {preco}")

def on_open(ws):
    print("🔌 Conectado ao WebSocket Binance.")

def on_error(ws, error):
    print(f"❌ Erro no WebSocket: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"🔌 Conexão encerrada. Código: {close_status_code}, Mensagem: {close_msg}")

def iniciar_websocket(symbol="pendleusdt"):
    ws = websocket.WebSocketApp(
        f"wss://stream.binance.com:9443/ws/{symbol}@trade",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

if __name__ == "__main__":
    iniciar_websocket("pendleusdt")