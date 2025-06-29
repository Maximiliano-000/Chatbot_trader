# 🧠 AnaliZ – Análises Técnicas Automatizadas com IA

O **AnaliZ** é um sistema completo para análise técnica automatizada de ativos financeiros, combinando **inteligência artificial**, **modelos preditivos (Prophet e LSTM)**, visualizações refinadas e geração de relatórios em PDF com qualidade profissional.

---

## 🚀 Funcionalidades Principais

✅ Previsões com **Prophet (séries temporais)**  
✅ Previsões com **redes neurais LSTM (TensorFlow)**  
✅ Interpretação com **GPT (OpenAI)**  
✅ Geração de **relatórios HTML e PDF premium**  
✅ Templates refinados com **modo escuro**, sumário automático e design responsivo  
✅ Histórico de previsões com banco **SQLite**  
✅ **Compartilhamento por WhatsApp**, integração PWA e UI leve em Flask  
✅ Setup automático para ambientes **macOS ARM (M1/M2)** com Conda  
✅ Suporte a automações com **APScheduler**

---

## 📁 Estrutura do Projeto

```
analyz/
├── app.py                  # Aplicação principal Flask
├── bot_trader.py          # Motor de análise técnica e predição
├── predict.py             # Predição com LSTM
├── train.py / train_cripto.py # Treinamento de modelos LSTM
├── model.py               # Modelos auxiliares (carregamento, salvamento, predição)
├── db.py                  # Banco de dados com Peewee (SQLite)
├── utils.py               # Funções auxiliares e geração de gráficos
├── templates/
│   ├── dashboard.html
│   ├── relatorio_custom.html
│   └── relatorio_premium.html
├── static/                # Logo, ícones e arquivos PWA
├── previsoes.db           # Banco com histórico de previsões LSTM
├── environment_macos.yml  # Ambiente Conda otimizado para M1/M2
├── requirements.txt       # Requisitos gerais do projeto
├── setup_analiz_definitivo.sh  # Script de instalação automatizada
└── ...
```

---

## 🖥️ Pré-visualização

### Relatório Premium com IA + Técnicos

![preview](https://user-images.githubusercontent.com/0000000/analyz_preview.png)

---

## ⚙️ Instalação Rápida (macOS com chip M1/M2)

```bash
# Clone o projeto
git clone https://github.com/seunome/analyz.git
cd analyz

# Crie e ative o ambiente Conda
conda env create -f environment_macos.yml
conda activate analiz_env

# Rode a aplicação
python app.py
```

---

## 📡 Tecnologias Utilizadas

- **Python 3.11**
- **Flask + Jinja2** – Backend e UI
- **Prophet / LSTM (TensorFlow)** – Previsão de séries temporais
- **OpenAI GPT** – Interpretação de dados
- **Peewee** – ORM leve para SQLite
- **WeasyPrint** – Exportação de relatórios para PDF
- **Plotly + Matplotlib** – Visualizações técnicas
- **APScheduler** – Automação de tarefas

---

## 📈 Exemplo de Uso

Acesse `http://localhost:5001/analise?ticker=WEGE3`  
Gere previsões LSTM e PDFs automaticamente para qualquer ativo (ações ou criptos).

---

## 📌 Roadmap Futuro

- [ ] Deploy com Docker
- [ ] Painel interativo com Streamlit
- [ ] Portal com login, planos e dashboards
- [ ] API REST pública para análise via token
- [ ] Testes com `pytest` + CI/CD

---

## 📄 Licença

© 2025 Wesley Maximiliano Braga. Todos os direitos reservados.  
Uso privado, acadêmico ou comercial mediante autorização prévia.

---

## 🤝 Contato

**Desenvolvido com dedicação por**  
📧 [wmaxib@yahoo.com.br](mailto:wmaxib@yahoo.com.br)  
🔗 [linkedin.com/in/wmaxib](https://linkedin.com/in/wmaxib)