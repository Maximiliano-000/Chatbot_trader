#!/bin/bash

echo "⚠️  Esse script irá limpar ambientes antigos, clonar o projeto e configurar tudo do zero."

read -p "Deseja continuar? (s/n): " confirm
if [[ "$confirm" != "s" ]]; then
  echo "❌ Cancelado."
  exit 0
fi

LOGFILE="setup_log_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -i $LOGFILE)
exec 2>&1

echo "🧼 Removendo ambientes antigos do Conda..."
conda env remove -n botenv -y 2>/dev/null
conda env remove -n botenv311 -y 2>/dev/null
conda env remove -n analiz_env -y 2>/dev/null
conda env remove -n analiz_arm -y 2>/dev/null
conda env remove -n analiz311 -y 2>/dev/null

echo "🧼 Removendo ambientes Pyenv antigos..."
sudo rm -rf ~/.pyenv/versions/botenv*
sudo rm -rf ~/.pyenv/versions/analiz*
sudo rm -rf ~/.pyenv/versions/chatbot*

echo "✅ Limpeza concluída."

echo "🚀 Criando ambiente Conda: analiz"
conda create -n analiz python=3.11 -y

echo "🔁 Ativando ambiente 'analiz'"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate analiz

echo "📦 Instalando TensorFlow (macOS ARM)"
pip install --upgrade pip setuptools wheel
pip install tensorflow-macos tensorflow-metal

echo "📦 Instalando outras dependências do projeto"
pip install -r requirements_macos_base.txt

echo "📊 Testando TensorFlow"
python -c "import tensorflow as tf; print('✅ TensorFlow instalado:', tf.__version__)"

# ⚙️ Clonagem do repositório (caso seja setup inicial em outro Mac)
if [[ ! -d "ChatbotTrader" ]]; then
  echo "📥 Clonando repositório ChatbotTrader..."
  git clone https://github.com/seu-usuario/ChatbotTrader.git
else
  echo "📁 Repositório já existe. Pulando clonagem."
fi

echo "✅ Tudo pronto! Log salvo em $LOGFILE"