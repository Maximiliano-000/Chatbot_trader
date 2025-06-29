import sqlite3
import pandas as pd

def obter_fluxo_ordens(ticker, limite=50):
    conn = sqlite3.connect('intraday.db')
    query = '''
        SELECT timestamp, price, quantity, side
        FROM order_flow
        WHERE ticker = ?
        ORDER BY timestamp DESC
        LIMIT ?
    '''
    df = pd.read_sql(query, conn, params=(ticker.upper(), limite))
    conn.close()
    return df