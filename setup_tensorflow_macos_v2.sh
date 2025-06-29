#!/bin/bash

echo "🔧 Iniciando setup TensorFlow para macOS ARM (Apple Silicon)"

# Verificação de arquitetura ARM
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
  echo "❌ Este script é compatível apenas com Macs com chip M1/M2 (ARM64)."
  exit 1
fi

# Nome do ambiente
ENV_NAME="analiz_env"

# Verifica se o Conda está instalado
if ! command -v conda &> /dev/null; then
    echo "❌ Conda não encontrado. Instale Miniforge ou Miniconda antes de continuar."
    exit 1
fi

# Remove ambiente antigo se existir
if conda env list | grep -q "$ENV_NAME"; then
  echo "⚠️ Ambiente "$ENV_NAME" já existe. Deseja removê-lo e recriar? (s/n)"
  read -r RESPOSTA
  if [[ "$RESPOSTA" == "s" ]]; then
    conda deactivate
    conda remove -n $ENV_NAME --all -y
  else
    echo "❌ Cancelado pelo usuário."
    exit 1
  fi
fi

# Criação do novo ambiente
echo "📦 Criando novo ambiente Conda "$ENV_NAME" com Python 3.10 (compatível)"
conda create -n $ENV_NAME python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Instala tensorflow-deps compatível + numpy correto
echo "⚙️ Instalando tensorflow-deps e numpy (Apple ARM compatível)"
conda install -c apple tensorflow-deps=2.10.0 numpy=1.23.3 -y

# Instala via pip os pacotes com aceleração Metal
echo "🚀 Instalando tensorflow-macos e tensorflow-metal via pip"
pip install --upgrade pip
pip install tensorflow-macos tensorflow-metal

# Verificação final
echo "🧪 Verificação final:"
python -c "import tensorflow as tf; print('✅ TensorFlow instalado:', tf.__version__)"

echo "✅ Ambiente "$ENV_NAME" configurado com sucesso para TensorFlow no macOS ARM."