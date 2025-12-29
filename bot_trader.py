# 🔧 Sistema e utilitários
import os
# Desabilitar GPU/Metal para evitar erros no Mac M1/M2
os.environ.setdefault("DYLD_LIBRARY_PATH", "/opt/homebrew/lib")
os.environ.setdefault("PKG_CONFIG_PATH", "/opt/homebrew/lib/pkgconfig")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduzir logs do TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Desabilitar CUDA

# Configurar TensorFlow para usar CPU
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Desabilitar GPU/Metal
except:
    pass

from datetime import datetime
from io import BytesIO

# 🔐 .env
from dotenv import load_dotenv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH, override=True)

print("ENV_PATH:", ENV_PATH)
print("TWELVE_DATA_API_KEY:", os.getenv("TWELVE_DATA_API_KEY"))

# 📊 Dados e análise
import numpy as np
import pandas as pd
from prophet import Prophet
from textblob import TextBlob
from scipy.stats import zscore

# 📈 Gráficos (sem statements aqui)
import plotly.express as px  # se a função de gráfico ainda existir/for usada

# 🤖 IA e APIs externas
import telebot

# 🌐 Web e interface Flask
from flask import Flask, request, jsonify, render_template, redirect, session, send_file
from markupsafe import Markup
from weasyprint import HTML

# 📆 Agendamentos
from apscheduler.schedulers.background import BackgroundScheduler

# 🧠 Módulos internos do projeto
from logger import uso_logger, erro_logger
from predict import prever_proximo_fechamento
from model import analise_com_gpt, analise_fallback, ajustar_previsao_lstm
from db import listar_previsoes, salvar_previsao

# 📁 Módulos internos em utils
from utils.financeiro import obter_dados, obter_dados_binance
from utils.indicadores import (
    calcular_indicadores,
    calcular_fibonacci,
    calcular_estrategia_longa,
    calcular_estrategia_short,
    validar_reversao_baixa,
    validar_reversao_alta,
    calcular_grau_confianca,
    gerar_microtendencia,
)
from utils.indicadores_avancados import calcular_adx, calcular_cci, calcular_vwap, calcular_atr
from utils.mensagem_estrategia import gerar_explicacao_estrategia, gerar_conclusao_dinamica
from utils.complementares import gerar_cenarios_alternativos
from utils.graficos import gerar_grafico
from utils.dados_com_fallback import obter_dados_com_fallback
from utils.moeda import detectar_mercado, moeda_por_mercado, simbolo_moeda, make_fmt
from utils.sinais import interpretar_sinais_tecnicos
from utils.fibonacci import interpretar_fibonacci_decisorio
from prophet_forecaster import executar_pipeline_completo
from lstm_forecaster import CriptoForecaster

def calcular_score_adaptativo(ticker, intervalo, periodo, dias=5):
    # Carrega dados usando yfinance com fallback Twelve Data
    dados = obter_dados(ticker, intervalo, periodo)

    # Previsões dos modelos
    previsao_prophet = executar_pipeline_completo(ticker, dados=dados, dias=dias)

    # Classifica as previsões
    classificacao_prophet = "Alta" if previsao_prophet['yhat'].iloc[-1] > dados['Close'].iloc[-1] else "Baixa"
    # Inicializa o objeto LSTM corretamente
    lstm_forecaster = CriptoForecaster(ticker=ticker, janela=60, epochs=10)

    # Verifica se o modelo existe, senão treina o modelo
    if lstm_forecaster.modelo_existente():
        lstm_forecaster.carregar_modelo_treinado()
    else:
        lstm_forecaster.carregar_dados()
        lstm_forecaster.treinar()

    # Faz a previsão corretamente usando o método 'prever'
    previsao_lstm = lstm_forecaster.prever(dias=dias)

    # Extrai o valor da última previsão feita pelo LSTM
    ultimo_previsao = previsao_lstm[-1]["valor"]

    # Obtém o preço atual (último preço)
    preco_atual = dados["Close"].iloc[-1]

    # Classifica corretamente a previsão LSTM
    classificacao_lstm = "Alta" if ultimo_previsao > preco_atual else "Baixa"

    # Converte classificação em scores numéricos
    score_prophet = {"Alta":1, "Baixa":-1}.get(classificacao_prophet, 0)
    score_lstm = {"Alta":1, "Baixa":-1}.get(classificacao_lstm, 0)

    # Interpretação dos indicadores técnicos
    indicadores_textuais = calcular_indicadores(dados)
    score_indicadores = interpretar_sinais_tecnicos(indicadores_textuais)

    from utils.ensemble import obter_pesos
    pesos = obter_pesos(intervalo)
    if pesos is None:
        raise ValueError(f"Intervalo '{intervalo}' não está definido nos pesos.")

    # Cálculo final do score adaptativo ponderado
    score_final = round(
        score_prophet * pesos["prophet"] +
        score_lstm * pesos["lstm"] +
        score_indicadores * pesos["indicadores"],
        2
    )
    print(f"[ENSEMBLE] Pesos usados para {ticker} ({intervalo}): {pesos} -> Score final: {score_final}")

    return score_final

# =============================================================================
# 1. Carregamento de variáveis de ambiente e configurações globais
# =============================================================================

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

bot = telebot.TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(24).hex()

# Lista de ativos que serão monitorados pelo scheduler
ativos_monitorados = [
    "MRFG3", "SUZB3", "EGIE3", "CMIG4", "JBSS3", "BRFS3",
    "WEGE3", "CSAN3", "POMO4", "COGN3", "CYRE3", "CSMG3",
    "SPSP3", "PSSA3", "HAPV3", "BBAS3", "ABEV3", "SOL-USD",
    "PENDLE-USD"
]

# =============================================================================
# 2. Variáveis de controle para estratégias de curto prazo
# =============================================================================
ultimo_status_alerta = {}
ultima_previsao_lstm = {}
acoes_compradas = {}
notificacoes = {}

# =============================================================================
# 4. Import da função de previsão LSTM e validador técnico
# =============================================================================

# Função de proteção para cálculos com fallback
def seguro(funcao, *args, **kwargs):
    try:
        resultado = funcao(*args, **kwargs)

        # Registro extra para atr ou outros valores técnicos
        nome = funcao.__name__
        if nome == "calcular_atr":
            import os
            import csv

            caminho = "logs_atr"
            os.makedirs(caminho, exist_ok=True)
            arquivo = os.path.join(caminho, "atr_logs.csv")

            linha = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                kwargs.get("ticker", "?"),
                round(resultado, 4) if isinstance(resultado, (int, float)) else str(resultado)
            ]

            with open(arquivo, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(linha)

        return resultado
    except Exception as e:
        print(f"[⚠️ ERRO seguro] {funcao.__name__}: {e}")
        return None

# =============================================================================
# 5. Estratégia de Curto Prazo (exemplo unificado)
# =============================================================================
def estrategia_curto_prazo(indicadores, ticker):
    """
    Exemplo de estratégia:
      - COMPRA se RSI < 30 (e ainda não houver compra).
      - VENDA se RSI > 70 ou se lucro >= 2%.
      - Envia notificação no Telegram.
      - Atualiza a previsão LSTM se a variação for maior que 0.5%.
    """
    global ultimo_status_alerta, ultima_previsao_lstm, acoes_compradas, notificacoes

    rsi_atual = indicadores['RSI'].iloc[-1]
    preco_atual = indicadores['Close'].iloc[-1]

    # Previsão LSTM
    proximo_valor_bruto = prever_proximo_fechamento(ticker)
    if proximo_valor_bruto is None:
        print(f"⚠️ Previsão não disponível para {ticker}.")
        return

    # Aplica ajuste técnico baseado na SMA20
    proximo_valor = ajustar_previsao_lstm(proximo_valor_bruto, indicadores)

    # Sinal de COMPRA
    if rsi_atual < 30:
        mensagem = f"📈 {ticker}: COMPRA forte (RSI={rsi_atual:.2f})"
        if ticker not in acoes_compradas and ultimo_status_alerta.get(ticker) != "COMPRA":
            notificacoes[ticker] = mensagem
            acoes_compradas[ticker] = preco_atual
            bot.send_message(CHAT_ID, mensagem)

    # Sinal de VENDA (se já comprou)
    elif ticker in acoes_compradas:
        preco_compra = acoes_compradas[ticker]
        lucro = (preco_atual - preco_compra) / preco_compra
        if rsi_atual > 70 or lucro >= 0.02:
            mensagem_venda = (
                f"📉 {ticker}: VENDA sugerida - Lucro: {lucro:.2%}, "
                f"Preço Atual: R${preco_atual:.2f}"
            )
            notificacoes[ticker] = mensagem_venda
            bot.send_message(CHAT_ID, mensagem_venda)
            del acoes_compradas[ticker]

    # Notificação de previsão LSTM (se variar mais que 0.5%)
    previsao_anterior = ultima_previsao_lstm.get(ticker)
    if previsao_anterior is None or abs(proximo_valor - previsao_anterior) / (previsao_anterior + 1e-10) > 0.005:
        mensagem_lstm = f"📊 Previsão LSTM para {ticker}: R$ {proximo_valor:.2f}"
        if proximo_valor != proximo_valor_bruto:
            mensagem_lstm += " ⚠️ (ajustada com base na SMA20)"
        notificacoes[ticker] = mensagem_lstm
        bot.send_message(CHAT_ID, mensagem_lstm)
        ultima_previsao_lstm[ticker] = proximo_valor

def prever(indicadores, dias=5, freq=None):
    """
    Previsão com Prophet ajustada para criptomoedas (sem fechamento diário).
    Frequência detectada automaticamente para suportar períodos como 15min, 30min, 1h etc.
    """

    try:
        df_prophet = indicadores.reset_index()

        # ✅ Ajuste robusto de data/hora
        if 'Datetime' in df_prophet.columns:
            df_prophet.rename(columns={'Datetime': 'ds'}, inplace=True)
        elif 'Date' in df_prophet.columns:
            df_prophet.rename(columns={'Date': 'ds'}, inplace=True)
        elif 'index' in df_prophet.columns:
            df_prophet.rename(columns={'index': 'ds'}, inplace=True)
        else:
            df_prophet['ds'] = indicadores.index

        if 'Close' not in df_prophet.columns:
            session["erro_prophet"] = "Coluna 'Close' ausente nos dados."
            return pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])

        df_prophet = df_prophet[['ds', 'Close']].dropna()
        df_prophet.rename(columns={'Close': 'y'}, inplace=True)
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

        freq_multiplicadores = {
            '15min': (4, '15min'),   # Próxima 1 hora (4 períodos)
            '30min': (4, '30min'),   # Próximas 2 horas
            '45min': (4, '45min'),   # Próximas 3 horas
            '1h': (6, '1H'),         # Próximas 6 horas
            '2h': (6, '2H'),         # Próximas 12 horas
            '6h': (4, '6H'),         # Próximas 24 horas
            '1d': (7, '1D'),         # Próximos 7 dias
            '5d': (4, '5D'),         # Próximos 20 dias
            '1m': (3, '30D')         # Próximos 90 dias (3 meses)
        }

        freq_detectada = pd.infer_freq(df_prophet['ds'].sort_values())

        # ✅ Seleção robusta da frequência
        if freq and freq.lower() in freq_multiplicadores:
            multiplicador, freq_final = freq_multiplicadores[freq.lower()]
        elif freq_detectada and freq_detectada.lower() in freq_multiplicadores:
            multiplicador, freq_final = freq_multiplicadores[freq_detectada.lower()]
        else:
            multiplicador, freq_final = (6, '1H')  # Default mais curto e coerente

        # ✅ Preencher série contínua
        df_prophet = df_prophet.set_index('ds').asfreq(freq_final, method='pad').fillna(method='ffill').reset_index()

        # ✅ Remoção leve de outliers
        z = np.abs(zscore(df_prophet['y']))
        df_temp = df_prophet[z < 3].copy()
        if df_temp.shape[0] >= 2:
            df_prophet = df_temp
        else:
            session["aviso_prophet"] = "Dados com outliers. Usando série completa."

        if df_prophet.shape[0] < 2:
            session["erro_prophet"] = "Dados insuficientes após filtragem."
            return pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])

        # ✅ Bollinger Bands
        sma = indicadores['Close'].rolling(20).mean().dropna()
        std = indicadores['Close'].rolling(20).std().dropna()
        sma20 = float(sma.iloc[-1]) if not sma.empty else indicadores['Close'].mean()
        std20 = float(std.iloc[-1]) if not std.empty else indicadores['Close'].std()
        bollinger_sup = sma20 + 2 * std20
        bollinger_inf = max(sma20 - 2 * std20, 0)

        df_prophet['cap'] = bollinger_sup
        df_prophet['floor'] = 0

        # ✅ Modelo Prophet
        modelo = Prophet(
            growth='logistic',
            changepoint_prior_scale=0.12,
            seasonality_mode='multiplicative',
            seasonality_prior_scale=10.0,
            interval_width=0.90
        )
        modelo.fit(df_prophet)

        total_periodos = dias * multiplicador

        futuro = modelo.make_future_dataframe(periods=total_periodos, freq=freq_final)
        futuro['cap'] = bollinger_sup
        futuro['floor'] = 0

        previsoes = modelo.predict(futuro)

        if previsoes.shape[0] < 2:
            session["erro_prophet"] = "Previsão insuficiente."
            return pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])

        # ✅ Suavização técnica
        previsoes['yhat'] = previsoes['yhat'] * 0.8 + sma20 * 0.2
        previsoes['yhat'] = previsoes['yhat'].clip(lower=bollinger_inf, upper=bollinger_sup)

        session['limite_minimo'] = round(bollinger_inf, 2)
        session['limite_maximo'] = round(bollinger_sup, 2)
        delta = previsoes['yhat'].iloc[-1] - previsoes['yhat'].iloc[0]
        session['viés_tendência'] = (
            "alta" if delta > sma20 * 0.01 else
            "baixa" if delta < -sma20 * 0.01 else
            "neutra"
        )
        session['ajuste_prophet'] = True

        return previsoes[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True)

    except Exception as e:
        session["erro_prophet"] = f"Erro Prophet: {str(e)}"
        return pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])

def gerar_visao_leiga_simplificada(tendencia):
    tendencia = tendencia.lower()
    if "alta" in tendencia:
        return "A IA entendeu que o mercado está otimista. Pode ser uma boa oportunidade, mas com consciência."
    elif "baixa" in tendencia:
        return "A IA identificou que o mercado está em queda. Isso exige cautela e avaliação estratégica antes de qualquer ação."
    elif "neutra" in tendencia:
        return "O mercado está estável no momento. A IA sugere observar mais antes de tomar decisões."
    else:
        return "A IA analisou os dados, mas não foi possível definir claramente se o momento é de alta ou de baixa. A recomendação é acompanhar com atenção."

# =============================================================================
# 7. Funções auxiliares de análise e gráficos
# =============================================================================

def sentimento_noticias(texto):
    """
    Exemplo de análise de sentimento de texto usando TextBlob.
    Retorna polaridade de -1 (negativo) a +1 (positivo).
    """
    blob = TextBlob(texto)
    return blob.sentiment.polarity

# =============================================================================
# 8. Rotas da API Flask
# =============================================================================
@app.route('/analise')
def analisar():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"erro": "Informe um ticker válido."}), 400

    dados = obter_dados(ticker)
    indicadores = calcular_indicadores(dados)

    # ✅ Validação robusta imediata após cálculo
    if indicadores.empty or "Close" not in indicadores.columns:
        erro_logger.error(f"⚠️ Indicadores não calculados corretamente para {ticker}. Verifique candles insuficientes ou coluna 'Close' ausente.")
        return jsonify({"erro": "Indicadores insuficientes ou coluna 'Close' ausente."}), 400

    previsao = prever(indicadores)
    analise = analise_com_gpt(ticker, indicadores, previsao)
    grafico = gerar_grafico(indicadores, ticker)
    
    recentes = session.get('recentes', [])
    if ticker not in recentes:
        recentes.insert(0, ticker)
        if len(recentes) > 5:
            recentes = recentes[:5]
        session['recentes'] = recentes

    uso_logger.info(f"Análise realizada para: {ticker} | IP: {request.remote_addr}")
    return render_template('dashboard.html', ticker=ticker, analise=analise, grafico=grafico, recentes=recentes)

def interpretar_indicadores(rsi, sma20, preco_atual, upper, lower):
    insights = []
    if rsi < 35:
        insights.append(f"<strong>RSI em {rsi}:</strong> A ação está entrando em zona de possível desconto. Pode ser oportunidade com cautela.")
    elif rsi > 65:
        insights.append(f"<strong>RSI em {rsi}:</strong> O ativo está próximo de sobrecompra. Pode haver resistência ou correção.")
    else:
        insights.append(f"<strong>RSI em {rsi}:</strong> O indicador está em uma zona neutra. O mercado não demonstra força clara no momento.")

    if preco_atual < sma20:
        insights.append("<strong>Preço abaixo da média de 20 dias:</strong> Indica tendência de baixa no curto prazo.")
    else:
        insights.append("<strong>Preço acima da média de 20 dias:</strong> Indica força no curto prazo.")

    if preco_atual <= lower:
        insights.append("<strong>Preço próximo da banda inferior:</strong> Pode ser sinal de possível fundo — mas cuidado com falsas esperanças.")
    elif preco_atual >= upper:
        insights.append("<strong>Preço próximo da banda superior:</strong> Pode estar esticado. Atenção a reversões.")

    return insights

def analisar_modelos_combinados(previsoes_lstm, previsoes_prophet):
    """
    Compara previsões LSTM (timing / curto prazo) e Prophet (estrutura / contexto).
    Retorna:
      - tendencia_combinada: string amigável para o relatório
      - tipo: rótulo semântico (Convergente / Multi-horizonte / Dados insuficientes)
      - media_ponderada: preço médio ponderado
      - direcao_lstm / direcao_prophet: para explicações curtas no template
      - leitura_mh_curta: texto explicativo curto quando for Multi-horizonte
    """
    base_insuf = {
        "tendencia_combinada": "Indefinida",
        "tipo": "Dados insuficientes",
        "media_ponderada": None,
        "direcao_lstm": None,
        "direcao_prophet": None,
        "leitura_mh_curta": ""
    }

    if not previsoes_lstm or not previsoes_prophet:
        return base_insuf

    # Últimas previsões (3 últimos pontos)
    try:
        lstm_vals = [float(p["valor"]) if isinstance(p, dict) else float(p) for p in previsoes_lstm[-3:]]
        prophet_vals = [float(p["yhat"]) for p in previsoes_prophet[-3:]]
    except Exception:
        return base_insuf

    if len(lstm_vals) < 2 or len(prophet_vals) < 2:
        return base_insuf

    # Médias
    media_lstm = sum(lstm_vals) / len(lstm_vals)
    media_prophet = sum(prophet_vals) / len(prophet_vals)

    # Média ponderada (ajustável)
    media_ponderada = round((0.6 * media_lstm + 0.4 * media_prophet), 2)

    # Direção (compara último com o primeiro do recorte)
    direcao_lstm = "alta" if lstm_vals[-1] > lstm_vals[0] else "baixa" if lstm_vals[-1] < lstm_vals[0] else "neutra"
    direcao_prophet = "alta" if prophet_vals[-1] > prophet_vals[0] else "baixa" if prophet_vals[-1] < prophet_vals[0] else "neutra"

    leitura_mh_curta = ""

    # Classificação semântica
    if direcao_lstm == "alta" and direcao_prophet == "alta":
        tipo = "Convergente (alta)"
        tendencia_combinada = "📈 Convergente (alta)"

    elif direcao_lstm == "baixa" and direcao_prophet == "baixa":
        tipo = "Convergente (baixa)"
        tendencia_combinada = "📉 Convergente (baixa)"

    else:
        tipo = "Multi-horizonte"
        tendencia_combinada = "🧭 Multi-horizonte (curto prazo vs estrutura)"

        leitura_mh_curta = (
            f"Curto prazo (LSTM): viés de {direcao_lstm}. "
            f"Estrutura (Prophet): {direcao_prophet}. "
            "Não há conflito entre modelos, mas diferença de horizonte temporal. "
            "O cenário favorece apenas operações curtas, seletivas e bem protegidas, "
            "enquanto operações direcionais exigem confirmação estrutural."
        )

    return {
        "tendencia_combinada": tendencia_combinada,
        "tipo": tipo,
        "media_ponderada": media_ponderada,
        "direcao_lstm": direcao_lstm,
        "direcao_prophet": direcao_prophet,
        "leitura_mh_curta": leitura_mh_curta
    }

@app.route("/previsao_custom")
def previsao_custom():
    from datetime import datetime
    import pandas as pd
    import math

    from utils.mensagem_estrategia import gerar_conclusao_dinamica
    from utils.dados_com_fallback import obter_preco_atual_binance
    from utils.fibonacci import interpretar_fibonacci_decisorio  # ajuste o caminho se estiver diferente

    # 🔐 Função universal de proteção numérica
    def seguro_float(valor):
        try:
            if isinstance(valor, dict):
                if "valor" in valor:
                    valor = valor["valor"]
                else:
                    valor = list(valor.values())[0]  # Fallback
            return round(float(valor), 2)
        except Exception:
            return 0.0

    # 🔒 Defaults de segurança (evita NameError / Pylance)
    media_ponderada = 0.0
    fibonacci = {}
    comentario_fibonacci = ""
    fibo_confluencia = False
    fibo_nivel = None
    fibo_tipo = None

    tipo_estrategia = "neutro"
    contexto = "indefinido"
    estrategia = {}
    previsoes_lstm = None

    # Parâmetros da URL
    ticker = request.args.get("ticker")
    periodo = request.args.get("periodo", "5d")

    if not ticker:
        return jsonify({"erro": "Ticker não informado."}), 400

    # [MOEDA] Detectar mercado/moeda e criar formatador
    mercado = detectar_mercado(ticker)
    moeda_sigla = moeda_por_mercado(mercado)     # 'USD' ou 'BRL'
    moeda_simbolo = simbolo_moeda(moeda_sigla)   # 'US$' ou 'R$'
    fmt = make_fmt(moeda_sigla)                  # usado no Fibonacci decisório

    # Intervalos
    periodos_map = {
        "1d": "1day", "2d": "1day", "3d": "1day", "5d": "1day", "7d": "1day",
        "15min": "15min", "30min": "30min", "45min": "30min",
        "1h": "1h", "2h": "1h", "6h": "1h",
        "1m": "1day", "3m": "1day", "6m": "1day", "1y": "1day"
    }

    intervalo_api = periodos_map.get(periodo, "1day")

    # ============================
    # 1) DADOS: Binance (primária) -> fallback (somente se vazio)
    # ============================
    dados = pd.DataFrame()
    intervalo_utilizado = intervalo_api
    mensagem_intervalo = None

    if ticker.endswith("-USD"):
        try:
            symbol = ticker.replace("-USD", "USDT")
            dados = obter_dados_binance(symbol=symbol, interval=intervalo_api, limit=130)
            intervalo_utilizado = intervalo_api
            mensagem_intervalo = None
        except Exception as e:
            uso_logger.warning(f"⚠️ Falha ao obter dados da Binance para {ticker}: {e}")
            dados = pd.DataFrame()

    # fallback apenas se veio vazio (independente da fonte)
    if dados.empty:
        uso_logger.warning(f"⚠️ Dados primários vazios para {ticker}, usando fallback.")
        dados, _, intervalo_utilizado, mensagem_intervalo = obter_dados_com_fallback(
            ticker=ticker,
            intervalo=intervalo_api,
            periodo=periodo,
            preferencia="twelve"
        )

    # Diagnóstico imediato
    if dados.empty or "Close" not in dados.columns or dados["Close"].dropna().empty:
        uso_logger.error(
            f"❌ Erro crítico para {ticker}. Dados vazios ou coluna Close insuficiente. "
            f"Colunas recebidas: {dados.columns.tolist()}, tamanho: {len(dados)}, "
            f"Close válidos: {dados['Close'].dropna().shape[0] if 'Close' in dados.columns else 0}, "
            f"Mensagem fallback: {mensagem_intervalo}."
        )
        return jsonify({
            "erro": "Dados insuficientes ou coluna 'Close' ausente.",
            "colunas_recebidas": dados.columns.tolist(),
            "tamanho_dos_dados": len(dados),
            "valores_validos_close": dados["Close"].dropna().shape[0] if "Close" in dados.columns else 0,
            "mensagem_fallback": mensagem_intervalo
        }), 400

    quantidade_candles = len(dados)

    dias_map = {"1day": 5, "15min": 90, "30min": 48, "1h": 24}
    dias = dias_map.get(intervalo_utilizado or intervalo_api, 5)

    if dados.empty:
        return render_template(
            "relatorio_custom.html",
            modo="html",
            erro_dados=True,
            ticker=ticker,
            periodo=periodo,
            quantidade_candles=quantidade_candles,
            datahora=datetime.now().strftime("%d/%m/%Y %H:%M"),
            aviso="",
            grafico=None,
            cenarios="",
            analise="",
            conclusao_final="",
            mensagem_intervalo=mensagem_intervalo,
            comentario_fibonacci="",
            moeda=moeda_simbolo
        )

    # ============================
    # 2) PROCESSAMENTO / RELATÓRIO
    # ============================
    try:
        indicadores = calcular_indicadores(dados, intervalo=intervalo_utilizado)

        vol = indicadores.get("Volume", pd.Series(dtype="float64"))
        indicadores["Volume"] = vol.ffill().bfill()

        if indicadores.empty or "Close" not in indicadores.columns or indicadores["Close"].dropna().empty:
            uso_logger.error(
                f"🛑 Problema em calcular_indicadores() para {ticker}. "
                f"vazio={indicadores.empty}, colunas={indicadores.columns.tolist()}"
            )
            return jsonify({
                "erro": "Indicadores insuficientes ou coluna 'Close' ausente após cálculo.",
                "indicadores_vazio": indicadores.empty,
                "colunas_recebidas": indicadores.columns.tolist(),
                "valores_validos_close": indicadores["Close"].dropna().shape[0] if "Close" in indicadores.columns else 0
            }), 400

        previsao = executar_pipeline_completo(
            ticker=ticker,
            dados=indicadores,
            dias=dias,
            freq=intervalo_utilizado
        )

        # Indicadores avançados (com seu wrapper "seguro")
        adx = seguro(calcular_adx, dados)
        cci = seguro(calcular_cci, dados)
        vwap = seguro(lambda x: calcular_vwap(x, silenciar=True), dados)
        atr_raw = seguro(calcular_atr, dados)

        # ATR normalizado (antes de TP/SL)
        if isinstance(atr_raw, dict):
            atr_val = atr_raw.get("valor") or 0.0
        else:
            atr_val = atr_raw
        try:
            atr_val = float(atr_val)
            if math.isnan(atr_val) or math.isinf(atr_val):
                atr_val = 0.0
        except Exception:
            atr_val = 0.0

        # Preço atual (binance se cripto -USD, senão do dataframe)
        preco_atual = seguro_float(indicadores["Close"].iloc[-1])
        if ticker.endswith("-USD"):
            try:
                symbol = ticker.replace("-USD", "USDT")
                preco_bin = obter_preco_atual_binance(symbol)
                preco_atual = preco_bin if preco_bin and preco_bin > 0 else preco_atual
            except Exception:
                pass

        # TP/SL sugeridos (ATR)
        tp_sugerido = round(preco_atual + (atr_val * 1.2), 4)
        sl_sugerido = round(preco_atual - (atr_val * 1.5), 4)

        analise = analise_com_gpt(ticker, indicadores, previsao)

        rsi = seguro_float(indicadores["RSI"].iloc[-1])
        sma20 = seguro_float(indicadores["SMA20"].iloc[-1])
        sma50 = seguro_float(indicadores["SMA50"].iloc[-1])

        upper_band = seguro_float(indicadores["UpperBand"].iloc[-1])
        lower_band = seguro_float(indicadores["LowerBand"].iloc[-1])
        volume_medio = seguro_float(indicadores["Volume_Medio"].iloc[-1])

        # Fibonacci (base)
        preco_max = float(indicadores["Close"].max())
        preco_min = float(indicadores["Close"].min())
        fibonacci = calcular_fibonacci(preco_min, preco_max)

        insights_tecnicos = interpretar_indicadores(rsi, sma20, preco_atual, upper_band, lower_band)

        conclusao_final = gerar_conclusao_dinamica(
            analise.get("tendencia", ""),
            rsi,
            preco_atual,
            sma20,
            moeda=moeda_simbolo
        )

        cenarios_df = gerar_cenarios_alternativos(preco_atual)
        cenarios_html = cenarios_df.to_html(index=False, classes="table", border=0)
        cenarios_dict = cenarios_df.to_dict(orient="records")

        print(f"Debug RSI: {rsi}, ADX: {adx}, CCI: {cci}, ATR: {atr_val}")

        # ============================
        # 3) LÓGICA DE ESTRATÉGIA
        # ============================
        intervalos_rigorosos = ["3h", "4h", "6h", "12h", "1d"]
        usar_seguranca = periodo in intervalos_rigorosos

        if usar_seguranca:
            if rsi < 40:
                if adx > 15 and preco_atual > sma20 and preco_atual > sma50:
                    estrategia = calcular_estrategia_longa(preco_atual, atr=atr_val, ticker=ticker, cenarios=cenarios_dict)
                    tipo_estrategia = "long"
                    contexto = "reversao confirmada robusta"
                elif adx > 15 and preco_atual > sma20:
                    estrategia = calcular_estrategia_longa(preco_atual, atr=atr_val, ticker=ticker, cenarios=cenarios_dict)
                    tipo_estrategia = "long"
                    contexto = "reversao confirmada"
                else:
                    estrategia = {}
                    tipo_estrategia = "neutro"
                    contexto = "reversao sem confirmação"

            elif rsi > 60:
                if adx > 15 and preco_atual < sma20 and preco_atual < sma50:
                    estrategia = calcular_estrategia_short(preco_atual, atr=atr_val, ticker=ticker, cenarios=cenarios_dict)
                    tipo_estrategia = "short"
                    contexto = "sobrecompra confirmada robusta"
                elif adx > 15 and preco_atual < sma20:
                    estrategia = calcular_estrategia_short(preco_atual, atr=atr_val, ticker=ticker, cenarios=cenarios_dict)
                    tipo_estrategia = "short"
                    contexto = "sobrecompra confirmada"
                else:
                    estrategia = {}
                    tipo_estrategia = "neutro"
                    contexto = "sobrecompra sem confirmação"

            elif cci < -80 and atr_val > 3:
                estrategia = calcular_estrategia_short(preco_atual, atr=atr_val, ticker=ticker, cenarios=cenarios_dict)
                tipo_estrategia = "short"
                contexto = "volatilidade"

            elif 50 < rsi <= 60 and preco_atual > sma20 and preco_atual > sma50:
                estrategia = calcular_estrategia_longa(preco_atual, atr=atr_val, ticker=ticker, cenarios=cenarios_dict)
                tipo_estrategia = "long"
                contexto = "forca robusta"

            elif 50 < rsi <= 60 and preco_atual > sma20:
                estrategia = calcular_estrategia_longa(preco_atual, atr=atr_val, ticker=ticker, cenarios=cenarios_dict)
                tipo_estrategia = "long"
                contexto = "forca"

            else:
                estrategia = calcular_estrategia_longa(preco_atual, atr=atr_val, ticker=ticker, cenarios=cenarios_dict)
                tipo_estrategia = "long"
                contexto = "neutro"

        else:
            if rsi < 40 and validar_reversao_alta(indicadores):
                estrategia = calcular_estrategia_longa(preco_atual, atr=atr_val, ticker=ticker, cenarios=cenarios_dict)
                tipo_estrategia = "long"
                contexto = "reversao técnica confirmada"

            elif rsi > 60 and validar_reversao_baixa(indicadores):
                estrategia = calcular_estrategia_short(preco_atual, atr=atr_val, ticker=ticker, cenarios=cenarios_dict)
                tipo_estrategia = "short"
                contexto = "sobrecompra confirmada"

            elif cci < -80 and atr_val > 3:
                estrategia = calcular_estrategia_short(preco_atual, atr=atr_val, ticker=ticker, cenarios=cenarios_dict)
                tipo_estrategia = "short"
                contexto = "volatilidade"

            elif 50 < rsi <= 60 and preco_atual > sma20 and preco_atual > sma50:
                estrategia = calcular_estrategia_longa(preco_atual, atr=atr_val, ticker=ticker, cenarios=cenarios_dict)
                tipo_estrategia = "long"
                contexto = "forca robusta"

            elif 50 < rsi <= 60 and preco_atual > sma20:
                estrategia = calcular_estrategia_longa(preco_atual, atr=atr_val, ticker=ticker, cenarios=cenarios_dict)
                tipo_estrategia = "long"
                contexto = "forca"

            else:
                estrategia = {}
                tipo_estrategia = "neutro"
                contexto = "sem confirmação técnica"

        # ============================
        # 4) LSTM + COMBINADA
        # ============================
        try:
            forecaster = CriptoForecaster(ticker, janela=60, epochs=100)
            if forecaster.modelo_existente():
                forecaster.carregar_modelo_treinado()
            else:
                forecaster.carregar_dados()
                forecaster.treinar()
            previsoes_lstm = forecaster.prever(dias=dias)
        except Exception:
            previsoes_lstm = None

        previsoes_prophet = previsao[["yhat"]].tail(3).to_dict(orient="records") if previsao is not None else []
        analise_combinada = analisar_modelos_combinados(previsoes_lstm, previsoes_prophet)
        media_ponderada = seguro_float(analise_combinada.get("media_ponderada", 0.0))
        preco_entrada = seguro_float(estrategia.get("preco_entrada", preco_atual))

        microtendencia = gerar_microtendencia(
            preco_atual=preco_entrada,
            previsoes_lstm=previsoes_lstm
        )

        # 🔍 Leitura Multi-Horizonte (substitui "divergente")
        if microtendencia and "alta" in microtendencia.lower() and "neutro" in analise_combinada.get("tipo", "").lower():
            leitura_multihorizonte = (
                "Curto prazo tenta alta (LSTM), enquanto a estrutura (Prophet) segue neutra. "
                "Movimento tático de curto prazo, sem confirmação estrutural."
        )
        elif microtendencia and "queda" in microtendencia.lower() and "neutro" in analise_combinada.get("tipo", "").lower():
            leitura_multihorizonte = (
                "Curto prazo pressiona para baixo (LSTM), com estrutura ainda neutra. "
                "Correção pontual dentro de um regime indefinido."
        )
        else:
            leitura_multihorizonte = (
                "Modelos operam em horizontes distintos, sem conflito direto. "
                "Aguardar confirmação para movimentos direcionais."
        )

        # ✅ Fibonacci decisório (única fonte de verdade)
        fibo_info = interpretar_fibonacci_decisorio(
            media_ponderada=media_ponderada,
            fibonacci=fibonacci,
            fmt_preco=fmt,
            tol_pct=0.0075
        )

        comentario_fibonacci = fibo_info.get("comentario", "")
        fibo_confluencia = bool(fibo_info.get("existe_confluencia", False))
        fibo_nivel = fibo_info.get("nivel")
        fibo_tipo = fibo_info.get("tipo")

        reversao_confirmada = (tipo_estrategia in ["short", "long"])

        grau_confiança = calcular_grau_confianca(
            tendencia_combinada=analise_combinada.get("tendencia_combinada", ""),
            microtendencia=microtendencia,
            reversao_confirmada=reversao_confirmada
        )

        estrategia_msg = gerar_explicacao_estrategia(
            tipo=tipo_estrategia,
            contexto=contexto,
            media_ponderada=media_ponderada,
            fibonacci=fibonacci,
            microtendencia=microtendencia,
            tendencia_combinada=analise_combinada.get("tendencia_combinada", ""),
            moeda=moeda_simbolo
        )

        avaliacao_estrategia = estrategia.get("avaliacao")

        if not estrategia:
            estrategia = {}
            tipo_estrategia = "neutro"
            contexto = "acompanhamento"
            estrategia_msg = gerar_explicacao_estrategia(
                tipo=tipo_estrategia,
                contexto=contexto,
                media_ponderada=media_ponderada,
                fibonacci=fibonacci,
                microtendencia=microtendencia,
                tendencia_combinada=analise_combinada.get("tendencia_combinada", ""),
                moeda=moeda_simbolo
            )
            avaliacao_estrategia = "Acompanhar movimentação técnica com atenção aos gatilhos."

        # consistência
        tipo_estrategia = (tipo_estrategia or "neutro").lower()

        if "fonte" not in locals():
            fonte = "desconhecida"
        if "intervalo_utilizado" not in locals():
            intervalo_utilizado = intervalo_api

        # ============================
        # 5) FLUXO DE ORDENS
        # ============================
        from utils.db import obter_fluxo_ordens
        dados_fluxo_intraday = obter_fluxo_ordens(ticker, limite=50)

        compras = 0.0
        vendas = 0.0
        if not dados_fluxo_intraday.empty and "side" in dados_fluxo_intraday.columns and "quantity" in dados_fluxo_intraday.columns:
            compras = dados_fluxo_intraday[dados_fluxo_intraday["side"] == "Compra"]["quantity"].sum()
            vendas = dados_fluxo_intraday[dados_fluxo_intraday["side"] == "Venda"]["quantity"].sum()

        if compras > vendas * 1.1:
            pressao = "compradora"
        elif vendas > compras * 1.1:
            pressao = "vendedora"
        else:
            pressao = "neutra"

        dados_fluxo_intraday_dict = dados_fluxo_intraday.to_dict(orient="records") if not dados_fluxo_intraday.empty else []

        # ============================
        # 6) RENDER
        # ============================
        return render_template(
            "relatorio_custom.html",
            ticker=ticker,
            periodo=periodo,
            datahora=datetime.now().strftime("%d/%m/%Y %H:%M"),

            rsi=rsi,
            sma20=sma20,
            sma50=sma50,
            upper_band=upper_band,
            lower_band=lower_band,
            volume_medio=volume_medio,

            grafico=gerar_grafico(indicadores, ticker, modo="html"),
            previsao=previsao[["ds", "yhat"]].tail().to_dict(orient="records") if previsao is not None else [],
            previsoes_lstm=previsoes_lstm,
            cenarios=Markup(cenarios_html),

            # Fibonacci decisório
            fibonacci=fibonacci or {},
            comentario_fibonacci=comentario_fibonacci,
            fibo_confluencia=fibo_confluencia,
            fibo_nivel=fibo_nivel,
            fibo_tipo=fibo_tipo,

            insights_tecnicos=insights_tecnicos,
            tendencia_combinada=analise_combinada.get("tendencia_combinada", ""),
            tipo=analise_combinada.get("tipo", ""),
            leitura_mh_curta=analise_combinada.get("leitura_mh_curta", ""),
            direcao_lstm=analise_combinada.get("direcao_lstm", ""),
            direcao_prophet=analise_combinada.get("direcao_prophet", ""),

            preco_entrada=preco_atual,
            microtendencia=microtendencia,
            media_ponderada=media_ponderada,
            leitura_multihorizonte=leitura_multihorizonte,

            explicacao_estrategia=estrategia_msg,
            tipo_estrategia=tipo_estrategia,
            avaliacao_estrategia=avaliacao_estrategia,
            grau_confianca=grau_confiança,

            tp_sugerido=tp_sugerido,
            sl_sugerido=sl_sugerido,
            tp1=seguro_float(estrategia.get("tp1")),
            tp2=seguro_float(estrategia.get("tp2")),
            tp3=seguro_float(estrategia.get("tp3")),
            sl=seguro_float(estrategia.get("sl")),

            moeda=moeda_simbolo,
            mensagem_intervalo=mensagem_intervalo,
            conclusao_final=conclusao_final,
            analise=analise.get("indicadores", ""),
            aviso=analise.get("aviso", ""),

            limite_minimo=session.get("limite_minimo", 0),
            limite_maximo=session.get("limite_maximo", 0),
            ajuste_prophet=session.get("ajuste_prophet", False),
            alerta_estabilidade=session.get("alerta_estabilidade", False),
            ia_falhou=session.get("ia_falhou", False),

            modo="html",
            fonte=fonte,
            intervalo_utilizado=intervalo_utilizado,
            dados_fluxo_intraday=dados_fluxo_intraday_dict,
            pressao=pressao
        )

    except Exception as e:
        erro_logger.error(f"Erro em /previsao_custom para {ticker}: {str(e)}")
        return jsonify({"erro": str(e)}), 500

@app.route('/analise_json')
def analise_json():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"erro": "Informe um ticker válido."}), 400

    try:
        dados = obter_dados(ticker)
        indicadores = calcular_indicadores(dados)
        previsao = prever(indicadores)

        # Análise com IA
        analise = analise_com_gpt(ticker, indicadores, previsao)

        # Gera gráfico Plotly (modo HTML ou JSON)
        grafico_fig = gerar_grafico(indicadores, ticker, modo='plotly')
        grafico_plotly_json = grafico_fig.to_json()

        return jsonify({
            "ticker": ticker,
            "analise": analise,
            "grafico_plotly": grafico_plotly_json
        })
    except Exception as e:
        erro_logger.error(f"Erro em /analise_json para {ticker}: {str(e)}")
        return jsonify({"erro": str(e)}), 500

@app.route('/exportar_pdf')
def exportar_pdf():
    ticker = request.args.get("ticker")
    if not ticker:
        return "Ticker não informado", 400

    try:
        dados = obter_dados(ticker)
        indicadores = calcular_indicadores(dados)
        previsao_df = prever(indicadores)

        try:
            analise = analise_com_gpt(ticker, indicadores, previsao_df)
            if not all(analise.get(chave) for chave in ["indicadores", "tendencia", "previsao", "estrategia", "risco"]):
                session["ia_falhou"] = True
                raise ValueError("IA retornou resposta incompleta.")
            else:
                session.pop("ia_falhou", None)
        except Exception as e:
            print(f"[ERRO GPT] {e}")
            session["ia_falhou"] = True
            analise = analise_fallback()

        # Indicadores técnicos
        preco_atual = float(indicadores['Close'].iloc[-1])
        rsi = round(indicadores['RSI'].iloc[-1], 2)
        sma20 = round(indicadores['SMA20'].iloc[-1], 2)
        sma50 = round(indicadores['SMA50'].iloc[-1], 2)
        upper = round(indicadores['UpperBand'].iloc[-1], 2)
        lower = round(indicadores['LowerBand'].iloc[-1], 2)
        volume_medio = round(indicadores['Volume_Medio'].iloc[-1], 2)
        conclusao_final = gerar_conclusao_dinamica(analise.get("tendencia", ""), rsi, preco_atual, sma20)
        insights_tecnicos = interpretar_indicadores(rsi, sma20, preco_atual, upper, lower)
        stop = round(preco_atual * 0.95, 2)
        alvo = round(preco_atual * 1.05, 2)

        # Gráfico em base64 para PDF
        grafico = gerar_grafico(indicadores, ticker, modo='base64')

        # Cenários alternativos
        cenarios_df = gerar_cenarios_alternativos(preco_atual)
        cenarios_html = cenarios_df.to_html(index=False, classes="table", border=0)

        # Valores previstos
        valores_previstos_raw = previsao_df['yhat'].tail(5)
        valores_filtrados = [max(0, v) for v in valores_previstos_raw]
        valores_html = "<ul>" + "".join([f"<li>R$ {v:.2f}</li>" for v in valores_filtrados]) + "</ul>"

        from lstm_forecaster import CriptoForecaster
        if "previsoes_lstm_cache" in session and ticker in session["previsoes_lstm_cache"]:
            previsoes_lstm = session["previsoes_lstm_cache"][ticker]
        else:
            forecaster = CriptoForecaster(ticker, janela=60, epochs=50)
            forecaster.carregar_dados()
            forecaster.treinar()
            previsoes_lstm = forecaster.prever(dias=5)
            previsoes_lstm = [float(v) for v in previsoes_lstm]  # 🔐 Conversão segura
            session.setdefault("previsoes_lstm_cache", {})[ticker] = previsoes_lstm

        # Histórico condicional por plano
        usuario = session.get("usuario")
        historico_html = ""
        if usuario:
            if usuario.get("plano") == "premium":
                previsoes = listar_previsoes(usuario.get("id"), limit=10)
                for p in previsoes:
                    historico_html += f"""
                    <tr>
                        <td>{p[0]}</td><td>{p[1]}</td><td>{p[2]}</td>
                        <td>R$ {float(p[3]):.2f}</td><td>{p[4]}</td>
                    </tr>
                    """
            elif usuario.get("plano") == "basico":
                historico_html = "<tr><td colspan='5'>Plano básico não inclui histórico. Faça upgrade.</td></tr>"
        else:
            historico_html = "<tr><td colspan='5'>Histórico disponível após login.</td></tr>"

        # Caminho absoluto para o banner
        caminho_banner = os.path.abspath("static/logo_banner.png")
        datahora = datetime.now().strftime('%d/%m/%Y %H:%M')  # ✅ Aqui

        # Renderiza HTML para PDF
        html_renderizado = render_template("relatorio_premium.html",
            ticker=ticker,
            rsi=rsi,
            sma20=sma20,
            sma50=sma50,
            upper_band=upper,
            lower_band=lower,
            volume_medio=volume_medio,
            tendencia=analise.get("tendencia", "-"),
            previsao=analise.get("previsao", "-"),
            estrategia=analise.get("estrategia", "-"),
            stop=stop,
            alvo=alvo,
            grafico=Markup(grafico),
            analise=analise,
            aviso=analise.get("aviso", ""),
            cenarios=Markup(cenarios_html),
            datahora=datahora,  # ✅ enviado para o template
            valores_previstos=Markup(valores_html),
            historico_tabela=Markup(historico_html),
            caminho_banner=caminho_banner,
            modo="pdf",
            insights_tecnicos=insights_tecnicos,
            conclusao_final=conclusao_final,
            dias=5,
            limite_minimo=session.get("limite_minimo", 0),
            limite_maximo=session.get("limite_maximo", 0),
            alerta_estabilidade=session.get("alerta_estabilidade", False),
            ajuste_prophet=session.get("ajuste_prophet", False),
            ajuste_lstm=session.get("ajuste_lstm", False),
            ia_falhou=session.get("ia_falhou", False),
            previsoes_lstm=previsoes_lstm,
            viés_tendência=session.get("viés_tendência", "-"),
            mensagem_prophet=session.get("mensagem_prophet", ""),
            aviso_sma20=session.get("aviso_sma20", None)
        )

        pdf = HTML(string=html_renderizado).write_pdf()

        for var in ['limite_minimo', 'limite_maximo', 'ajuste_prophet', 'alerta_estabilidade', 'ajuste_lstm']:
            session.pop(var, None)

        return send_file(BytesIO(pdf),
                         download_name=f"AnaliZ_{ticker}_relatorio.pdf",
                         as_attachment=True,
                         mimetype='application/pdf')

    except Exception as e:
        erro_logger.error(f"Erro em /exportar_pdf para {ticker}: {str(e)}")
        return f"Erro ao gerar PDF: {str(e)}", 500

@app.route("/exportar_pdf_custom")
def exportar_pdf_custom():
    from lstm_forecaster import CriptoForecaster
    from weasyprint import HTML
    from io import BytesIO

    ticker = request.args.get("ticker")
    periodo = request.args.get("periodo", "5d")
    dias_map = {"1d": 1, "5d": 5, "6mo": 130, "1y": 260, "15min": 15, "30min": 30, "45min": 45, "60min": 60}
    dias = dias_map.get(periodo, 5)

    try:
        dados = obter_dados(ticker, intervalo=periodo)
        indicadores = calcular_indicadores(dados)
        previsao = prever(indicadores, dias=dias)

        analise = analise_com_gpt(ticker, indicadores, previsao)
        grafico = gerar_grafico(indicadores, ticker, modo="base64")

        rsi = round(indicadores['RSI'].iloc[-1], 2)
        sma20 = round(indicadores['SMA20'].iloc[-1], 2)
        sma50 = round(indicadores['SMA50'].iloc[-1], 2)
        upper_band = round(indicadores['UpperBand'].iloc[-1], 2)
        lower_band = round(indicadores['LowerBand'].iloc[-1], 2)
        volume_medio = round(indicadores['Volume_Medio'].iloc[-1], 2)
        preco_atual = round(indicadores['Close'].iloc[-1], 2)
        conclusao_final = gerar_conclusao_dinamica(analise.get("tendencia", ""), rsi, indicadores['Close'].iloc[-1], sma20)
        insights_tecnicos = interpretar_indicadores(rsi, sma20, indicadores['Close'].iloc[-1], upper_band, lower_band)
        cenarios_df = gerar_cenarios_alternativos(indicadores['Close'].iloc[-1])
        cenarios_html = cenarios_df.to_html(index=False, classes="table", border=0)
        cenarios_dict = cenarios_df.to_dict(orient="records")

        if rsi > 70:
            _ = calcular_estrategia_short(preco_atual, cenarios=cenarios_dict)
        else:
            _ = calcular_estrategia_longa(preco_atual, cenarios=cenarios_dict)

        try:
            forecaster = CriptoForecaster(ticker, janela=60, epochs=50)
            forecaster.carregar_dados()
            forecaster.treinar()
            previsoes_lstm = forecaster.prever(dias=5)
            previsoes_lstm = [float(v) for v in previsoes_lstm]
        except Exception:
            previsoes_lstm = None

        caminho_banner = os.path.abspath("static/logo_banner.png")
        html_renderizado = render_template("relatorio_custom.html",
            ticker=ticker,
            periodo=periodo,
            datahora=datetime.now().strftime('%d/%m/%Y %H:%M'),
            rsi=rsi,
            sma20=sma20,
            sma50=sma50,
            upper_band=upper_band,
            lower_band=lower_band,
            volume_medio=volume_medio,
            previsao=previsao[['ds', 'yhat']].tail().to_dict(orient='records'),
            previsoes_lstm=previsoes_lstm,
            cenarios=Markup(cenarios_html),
            insights_tecnicos=insights_tecnicos,
            conclusao_final=conclusao_final,
            limite_minimo=session.get("limite_minimo", 0),
            limite_maximo=session.get("limite_maximo", 0),
            ajuste_prophet=session.get("ajuste_prophet", False),
            alerta_estabilidade=session.get("alerta_estabilidade", False),
            analise=analise.get("indicadores", ""),
            aviso=analise.get("aviso", ""),
            modo="pdf",
            ia_falhou=session.get("ia_falhou", False),
            grafico=grafico,
            caminho_banner=caminho_banner
        )

        pdf = HTML(string=html_renderizado).write_pdf()

        return send_file(BytesIO(pdf),
                         download_name=f"AnaliZ_Custom_{ticker}_{periodo}.pdf",
                         as_attachment=True,
                         mimetype='application/pdf')

    except Exception as e:
        erro_logger.error(f"Erro ao gerar PDF customizado para {ticker}: {str(e)}")
        return f"Erro ao gerar PDF customizado: {str(e)}", 500

@app.route('/')
def home():
    return '''
    <h2>Bot Trader Ativo ✅</h2>
    <p>Use <code>/analise?ticker=WEGE3</code> para acessar uma análise completa.</p>
    '''
# =============================================================================
# 10. Configuração do Scheduler (tarefas agendadas)
# =============================================================================

scheduler = BackgroundScheduler()

# Exemplo (opcional): agendamento de relatório diário
# scheduler.add_job(func=relatorio_periodico, trigger="cron", hour=9, minute=30)

scheduler.start()

@app.route("/relatorio")
def relatorio():

    ticker = request.args.get("ticker")
    # [MOEDA] Detectar mercado/moeda e criar formatador
    mercado = detectar_mercado(ticker)
    moeda_sigla = moeda_por_mercado(mercado)
    moeda_simbolo = simbolo_moeda(moeda_sigla)
    fmt = make_fmt(moeda_sigla)

    usuario = session.get("usuario")

    if not ticker:
        return jsonify({"erro": "Informe um ticker válido."}), 400

    try:
        dados = obter_dados(ticker)
    except ValueError as e:
        return render_template("erro.html", mensagem=str(e))

    indicadores = calcular_indicadores(dados)

    if indicadores.empty or len(indicadores) < 2:
        rsi = round(indicadores["RSI"].iloc[-1], 2)
        sma20 = round(indicadores["SMA20"].iloc[-1], 2)
        sma50 = round(indicadores["SMA50"].iloc[-1], 2)

    previsao_df = prever(indicadores)

    try:
        analise = analise_com_gpt(ticker, indicadores, previsao_df)
        if not all(analise.get(k) for k in ["indicadores", "tendencia", "previsao", "estrategia", "risco"]):
            session["ia_falhou"] = True
            raise ValueError("IA retornou resposta incompleta.")
        else:
            session.pop("ia_falhou", None)
    except Exception as e:
        print(f"[ERRO GPT] {e}")
        session["ia_falhou"] = True
        analise = analise_fallback()

    def extrair(df, col):
        try:
            return round(float(df[col].iloc[-1]), 2)
        except Exception:
            return 0

    preco_atual = extrair(indicadores, "Close")
    rsi = extrair(indicadores, "RSI")
    sma20 = extrair(indicadores, "SMA20")
    sma50 = extrair(indicadores, "SMA50")
    upper = extrair(indicadores, "UpperBand")
    lower = extrair(indicadores, "LowerBand")
    volume_medio = extrair(indicadores, "Volume_Medio")

    try:
        conclusao_final = gerar_conclusao_dinamica(analise.get("tendencia", ""), rsi, preco_atual, sma20)
    except Exception:
        conclusao_final = "Não foi possível gerar a conclusão técnica com os dados disponíveis."

    try:
        stop = round(preco_atual * 0.95, 2) if preco_atual else 0
        alvo = round(preco_atual * 1.05, 2) if preco_atual else 0
    except Exception:
        stop = alvo = 0

    try:
        grafico = gerar_grafico(indicadores, ticker)
    except Exception as e:
        print(f"[ERRO GRAFICO] {e}")
        grafico = ""

    cenarios_df = gerar_cenarios_alternativos(preco_atual)
    cenarios_html = cenarios_df.to_html(index=False, classes="table", border=0)

    valores_previstos_raw = previsao_df['yhat'].tail(5)
    valores_filtrados = [max(0, v) for v in valores_previstos_raw]
    valores_html = "<ul>" + "".join([f"<li>{fmt(v)}</li>" for v in valores_filtrados]) + "</ul>"

    historico_html = ""
    if usuario:
        if usuario.get("plano") == "premium":
            previsoes = listar_previsoes(usuario.get("id"), limit=10)
            for p in previsoes:
                historico_html += f"""
                <tr><td>{p[0]}</td><td>{p[1]}</td><td>{p[2]}</td>
                <td>{fmt(float(p[3]))}</td><td>{p[4]}</td></tr>
                """
        elif usuario.get("plano") == "basico":
            historico_html = "<tr><td colspan='5'>Plano básico não inclui histórico. <a href='#'>Faça upgrade.</a></td></tr>"
    else:
        historico_html = "<tr><td colspan='5'>Histórico disponível após login.</td></tr>"

    # ✅ Previsão LSTM com cache por ticker
    from lstm_forecaster import CriptoForecaster
    if "previsoes_lstm_cache" in session and ticker in session["previsoes_lstm_cache"]:
        previsoes_lstm = session["previsoes_lstm_cache"][ticker]
    else:
        try:
            forecaster = CriptoForecaster(ticker, janela=60, epochs=50)
            forecaster.carregar_dados()
            forecaster.treinar()
            previsoes_lstm = forecaster.prever(dias=5)
            # Converte os valores para float (tipo nativo Python) antes de salvar na sessão
            previsoes_lstm = [float(v) for v in previsoes_lstm]
            session.setdefault("previsoes_lstm_cache", {})[ticker] = previsoes_lstm

        except Exception as e:
            print(f"[LSTM ERRO] {e}")
            previsoes_lstm = None

    session.pop("lstm_credito_usado", None)
    uso_logger.info(f"Relatório premium gerado para: {ticker} | IP: {request.remote_addr}")

    datahora = datetime.now().strftime('%d/%m/%Y %H:%M')

    return render_template("relatorio_premium.html",
        ticker=ticker,
        rsi=rsi,
        sma20=sma20,
        sma50=sma50,
        upper_band=upper,
        lower_band=lower,
        volume_medio=volume_medio,
        tendencia=analise.get("tendencia", "-"),
        previsao=analise.get("previsao", "-"),
        estrategia=analise.get("estrategia", "-"),
        stop=stop,
        alvo=alvo,
        grafico=grafico,
        analise=analise,
        aviso=analise.get("aviso", ""),
        cenarios=Markup(cenarios_html),
        datahora=datahora,
        valores_previstos=Markup(valores_html),
        historico_tabela=Markup(historico_html),
        ajuste_lstm=session.get("ajuste_lstm", False),
        ajuste_prophet=session.get("ajuste_prophet", False),
        alerta_estabilidade=session.get("alerta_estabilidade", False),
        limite_minimo=session.get("limite_minimo", 0),
        limite_maximo=session.get("limite_maximo", 0),
        dias=5,
        ia_falhou=session.get("ia_falhou", False),
        modo="html",
        previsoes_lstm = None,  # ← não carregamos LSTM por padrão
        viés_tendência=session.get("viés_tendência", "-"),
        mensagem_prophet=session.get("mensagem_prophet", ""),
        aviso_sma20=session.get("aviso_sma20", None),
        visao_leiga=analise.get("visao_leiga", None),
        insights_tecnicos=interpretar_indicadores(rsi, sma20, preco_atual, upper, lower),
        conclusao_final=conclusao_final
    )

@app.route("/login_simulado")
def login_simulado():
    # ⚠️ Apenas para testes! Em produção, use autenticação real
    session["usuario"] = {"id": "usuario123", "plano": "premium"}
    return '''
    <h3>Login simulado realizado com sucesso!</h3>
    <p>Você está autenticado como <strong>usuário premium</strong>.</p>
    <p>Agora pode acessar <code>/prever_lstm?ticker=WEGE3</code>.</p>
    '''

@app.route("/planos")
def planos():
    return render_template("planos.html")

@app.route("/comprar_credito")
def comprar_credito():
    usuario = session.get("usuario")
    if not usuario:
        return redirect("/login_simulado")  # ou página de login real

    # Adiciona 1 crédito ao usuário
    usuario["creditos"] = usuario.get("creditos", 0) + 1
    session["usuario"] = usuario

    # Redireciona para a página de sucesso visual
    return render_template("sucesso.html")

@app.route("/painel")
def painel():
    return render_template("painel.html")

@app.route("/turbinar_lstm")
def turbinar_lstm():
    ticker = request.args.get("ticker")
    if not ticker:
        return "Ticker não informado", 400

    usuario = session.get("usuario")
    if not usuario:
        return redirect("/login_simulado")  # ou /login

    # Verifica plano ou crédito
    plano = usuario.get("plano")
    creditos = usuario.get("creditos", 0)

    if plano != "premium" and creditos < 1:
        return render_template("erro.html", mensagem="Você precisa de créditos ou um plano premium para acessar a previsão LSTM.")

    # Se for por crédito, desconta 1
    if plano != "premium" and creditos >= 1:
        usuario["creditos"] -= 1
        session["usuario"] = usuario  # atualiza session

    # Gera o relatório com LSTM e redireciona para o PDF
    return redirect(f"/exportar_pdf?ticker={ticker}")

@app.route("/prever_lstm")
def prever_lstm():
    ticker = request.args.get("ticker")
    period = request.args.get("period", "1y")
    janela = int(request.args.get("janela", 20))

    if not ticker:
        return jsonify({"erro": "Ticker não informado"}), 400

    usuario = session.get("usuario")
    if not usuario or usuario.get("plano") != "premium":
        return jsonify({"erro": "Acesso restrito ao plano Premium"}), 403

    user_id = usuario.get("id")
    ticker_yf = ticker if "-USD" in ticker else f"{ticker}.SA"

    try:
        dados = obter_dados(ticker_yf)
        indicadores = calcular_indicadores(dados)

        valor_original = prever_proximo_fechamento(ticker_yf, janela=janela, period=period)
        valor = ajustar_previsao_lstm(valor_original, indicadores)
        valor = float(valor)

        if valor != valor_original:
            uso_logger.info(f"[LSTM Ajustado] {ticker}: {valor_original:.2f} → {valor:.2f}")

    except Exception as e:
        erro_logger.error(f"Erro em /prever_lstm para {ticker}: {str(e)}")
        return jsonify({"erro": f"Erro interno: {str(e)}"}), 500

    if valor is None:
        return jsonify({"erro": f"Não foi possível prever {ticker}"}), 500

    salvar_previsao(user_id, ticker, period, janela, valor)

    moeda = "US$" if "-USD" in ticker else "R$"
    return jsonify({
        "ticker": ticker,
        "previsao": round(valor, 2),
        "moeda": moeda
    })

@app.route("/prever_lstm_redirecionar")
def prever_lstm_redirecionar():
    ticker = request.args.get("ticker")
    period = "1y"
    janela = 20

    if not ticker:
        return "Ticker não informado", 400

    # 🔐 Verifica plano
    usuario = session.get("usuario")
    if not usuario or usuario.get("plano") != "premium":
        return "Acesso restrito ao plano Premium", 403

    try:
        ticker_yf = ticker if "-USD" in ticker else f"{ticker}.SA"
        valor = prever_proximo_fechamento(ticker_yf, janela=janela, period=period)
        if valor:
            salvar_previsao(usuario["id"], ticker, period, janela, float(valor))
            return redirect(f"/exportar_pdf?ticker={ticker}")
        else:
            return f"Não foi possível gerar previsão para {ticker}", 500
    except Exception as e:
        erro_logger.error(f"[REDIRECIONAR] Erro: {e}")
        return f"Erro ao gerar previsão: {str(e)}", 500

@app.route("/historico_previsoes")
def historico_previsoes():
    usuario = session.get("usuario")
    if not usuario or usuario.get("plano") != "premium":
        return jsonify({"erro": "Acesso restrito ao plano Premium"}), 403

    user_id = usuario.get("id")
    try:
        historico = listar_previsoes(user_id)
        # Serializa os dados para JSON
        dados = [
            {
                "ticker": h[0],
                "period": h[1],
                "janela": h[2],
                "previsao": round(float(h[3]), 2),
                "data_hora": h[4]
            }
            for h in historico
        ]
        return jsonify(dados)
    except Exception as e:
        erro_logger.error(f"Erro ao listar histórico de previsões: {str(e)}")
        return jsonify({"erro": "Erro interno ao consultar histórico."}), 500

def teste_grafico_express():
    # Criando dados de teste
    datas = pd.date_range(start="2024-12-01", periods=60)
    preco = np.random.normal(20, 1, size=60)
    sma20 = pd.Series(preco).rolling(20).mean()
    sma50 = pd.Series(preco).rolling(50).mean()

    df = pd.DataFrame({
        "Data": datas,
        "Preço": preco,
        "SMA20": sma20,
        "SMA50": sma50,
    })

    # Plot usando plotly express
    fig = px.line(df, x="Data", y=["Preço", "SMA20", "SMA50"], title="Teste de Linhas – plotly.express")
    fig.update_layout(template="plotly_dark", height=400)

    fig.show()

# =============================================================================
# 11. Execução da aplicação Flask
# =============================================================================
if __name__ == "__main__":
    try:
        app.run(debug=True, port=5001)
    finally:
        scheduler.shutdown()