# db.py
import sqlite3
from datetime import datetime
import os

# Garante que a pasta de dados exista
os.makedirs("dados", exist_ok=True)

DB_PATH = "dados/previsoes.db"

def conectar():
    return sqlite3.connect(DB_PATH)

def criar_tabela():
    conn = conectar()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS previsoes_lstm (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        ticker TEXT NOT NULL,
        period TEXT,
        janela INTEGER,
        previsao REAL,
        data_hora DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def salvar_previsao(user_id, ticker, period, janela, previsao):
    """
    Salva uma previsão LSTM no banco de dados.
    """
    conn = conectar()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO previsoes_lstm (user_id, ticker, period, janela, previsao, data_hora)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, ticker, period, janela, previsao, datetime.now()))
    conn.commit()
    conn.close()

def listar_previsoes(user_id, limit=20):
    """
    Lista as previsões recentes de um usuário.
    """
    conn = conectar()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT ticker, period, janela, previsao, data_hora
    FROM previsoes_lstm
    WHERE user_id = ?
    ORDER BY data_hora DESC
    LIMIT ?
    """, (user_id, limit))
    resultados = cursor.fetchall()
    conn.close()
    return resultados