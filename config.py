import os
from dotenv import load_dotenv
load_dotenv()

print("KEY:", os.getenv("TWELVE_DATA_API_KEY"))  # Deve imprimir sua chave

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")