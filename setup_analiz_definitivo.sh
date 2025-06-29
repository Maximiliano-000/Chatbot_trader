#!/bin/bash

echo "🔧 Criando ambiente Conda: analiz_arm"
conda create -n analiz_arm python=3.11 -y

echo "📦 Ativando Conda e o ambiente"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate analiz_arm

echo "🍎 Instalando bibliotecas gráficas do sistema (necessárias para WeasyPrint)"
brew install cairo pango gdk-pixbuf libffi gobject-introspection

echo "📦 Instalando pacotes essenciais via conda"
conda install -c conda-forge \
  pandas numpy matplotlib yfinance flask jinja2 werkzeug itsdangerous markupsafe \
  prophet openai textblob plotly apscheduler peewee python-dotenv weasyprint scikit-learn -y

echo "📥 Instalando pacotes via pip (TensorFlow + Telegram)"
pip install tensorflow-macos tensorflow-metal pytelegrambotapi

echo "✅ Verificando TensorFlow"
python -c "import tensorflow as tf; print('TensorFlow OK:', tf.__version__)"

echo "✅ Verificando WeasyPrint"
python -c "from weasyprint import HTML; print('WeasyPrint funcionando!')"

echo "🎉 Ambiente 'analiz_arm' configurado com sucesso!"