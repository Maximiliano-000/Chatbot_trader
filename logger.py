# logger.py
import logging
import os

LOG_DIR = "analiZ/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log de uso
uso_logger = logging.getLogger("uso")
uso_handler = logging.FileHandler(f"{LOG_DIR}/uso.log", encoding='utf-8')
uso_formatter = logging.Formatter('%(asctime)s - %(message)s')
uso_handler.setFormatter(uso_formatter)
uso_logger.setLevel(logging.INFO)
uso_logger.addHandler(uso_handler)

# Log de erro
erro_logger = logging.getLogger("erros")
erro_handler = logging.FileHandler(f"{LOG_DIR}/erros.log", encoding='utf-8')
erro_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
erro_handler.setFormatter(erro_formatter)
erro_logger.setLevel(logging.ERROR)
erro_logger.addHandler(erro_handler)