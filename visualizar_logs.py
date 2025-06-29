# visualizar_logs.py
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Logs AnaliZ", layout="wide")
st.title("📋 Visualizador de Logs – AnaliZ")

LOG_DIR = "analiZ/logs"
uso_log_path = os.path.join(LOG_DIR, "uso.log")
erro_log_path = os.path.join(LOG_DIR, "erros.log")

def carregar_log(caminho, tipo):
    if not os.path.exists(caminho):
        return pd.DataFrame()
    
    with open(caminho, encoding="utf-8") as f:
        linhas = f.readlines()
    
    registros = []
    for linha in linhas:
        if " - " in linha:
            partes = linha.strip().split(" - ", maxsplit=2)
            if tipo == "uso":
                registros.append({"Data/Hora": partes[0], "Descrição": partes[1]})
            elif tipo == "erro" and len(partes) == 3:
                registros.append({"Data/Hora": partes[0], "Nível": partes[1], "Mensagem": partes[2]})
    return pd.DataFrame(registros)

aba = st.sidebar.radio("📁 Selecionar log", ["Log de Uso", "Log de Erros"])

if aba == "Log de Uso":
    df = carregar_log(uso_log_path, tipo="uso")
    st.subheader("📈 Análises Realizadas")
    if not df.empty:
        filtro = st.text_input("🔍 Filtrar por texto ou ticker:")
        if filtro:
            df = df[df["Descrição"].str.contains(filtro, case=False)]
        st.dataframe(df[::-1], use_container_width=True)
    else:
        st.warning("Nenhum dado encontrado em uso.log.")

elif aba == "Log de Erros":
    df = carregar_log(erro_log_path, tipo="erro")
    st.subheader("🚨 Erros do Sistema")
    if not df.empty:
        filtro = st.text_input("🔍 Filtrar por palavra-chave:")
        if filtro:
            df = df[df["Mensagem"].str.contains(filtro, case=False)]
        st.dataframe(df[::-1], use_container_width=True)
    else:
        st.success("Nenhum erro registrado recentemente. Sistema estável! ✅")