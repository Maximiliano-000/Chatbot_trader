import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="AnaliZ – Trader Inteligente", layout="wide")

def texto_justificado(texto):
    st.markdown(f"""
        <div style='text-align: justify; font-size: 1.1em;'>
            {texto}
        </div>
    """, unsafe_allow_html=True)

st.title("🤖 AnaliZ • Trader Inteligente")

opcoes = [
    "MRFG3", "SUZB3", "EGIE3", "CMIG4", "JBSS3", "BRFS3", "WEGE3", "CSAN3",
    "POMO4", "COGN3", "CYRE3", "CSMG3", "SPSP3", "PSSA3", "HAPV3",
    "BBAS3", "ABEV3", "SOL-USD", "PENDLE-USD"
]

ticker = st.selectbox("📌 Escolha o ticker da ação:", opcoes)

tab1, tab2 = st.tabs(["📊 Análise Técnica", "🔮 Previsão com LSTM"])

with tab1:
    if st.button("Gerar Análise Técnica"):
        with st.spinner("🔎 Gerando análise técnica..."):
            try:
                resposta = requests.get(f"http://127.0.0.1:5001/analise_json?ticker={ticker.upper()}")
                resposta.raise_for_status()
                resultado = resposta.json()

                analise = resultado.get('analise', {})
                grafico_plotly = resultado.get('grafico_plotly')

                st.subheader(f"📈 Resultado para {ticker.upper()}:")
                texto_justificado(analise.get("indicadores", "Análise indisponível."))

                if grafico_plotly:
                    fig = go.Figure().from_json(grafico_plotly)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Gráfico Plotly não disponível.")
            except requests.exceptions.RequestException as e:
                st.error(f"Erro ao conectar com a API: {e}")

with tab2:
    if st.button("Prever com LSTM e Salvar Histórico"):
        with st.spinner("🔮 Gerando previsão com LSTM..."):
            try:
                resposta = requests.get(f"http://127.0.0.1:5001/prever_lstm?ticker={ticker.upper()}&period=6mo&janela=20")
                resposta.raise_for_status()
                resultado = resposta.json()
                valor = resultado.get('previsao')
                if valor:
                    moeda = resultado.get("moeda", "R$")
                    st.success(f"📍 Previsão LSTM para {ticker.upper()}: {moeda} {valor:.2f}")
                else:
                    st.error(f"Não foi possível obter a previsão para {ticker.upper()}.")
            except requests.exceptions.RequestException as e:
                st.error(f"Erro ao obter previsão: {e}")