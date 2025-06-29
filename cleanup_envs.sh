#!/bin/bash

echo "⚠️  Esse script apagará ambientes Python obsoletos do sistema."

read -p "Deseja continuar? (s/n): " confirm
if [[ "$confirm" != "s" ]]; then
  echo "❌ Cancelado."
  exit 0
fi

echo "🧼 Limpando ambientes Conda antigos..."
sudo rm -rf ~/miniconda3/envs/botenv*
sudo rm -rf ~/miniconda3/envs/analiz*

echo "🧼 Limpando ambientes Pyenv antigos..."
sudo rm -rf ~/.pyenv/versions/botenv*
sudo rm -rf ~/.pyenv/versions/analiz*
sudo rm -rf ~/.pyenv/versions/chatbot*

echo "✅ Limpeza finalizada. Reabra o VS Code para atualizar os ambientes."