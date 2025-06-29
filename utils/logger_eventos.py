import csv
import os
from datetime import datetime

def registrar_evento_fallback(ticker, intervalo, mensagem, caminho="dados/log_fallback.csv"):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    registro = {
        "data_hora": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ticker": ticker,
        "intervalo_utilizado": intervalo,
        "mensagem": mensagem
    }

    with open(caminho, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=registro.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(registro)
