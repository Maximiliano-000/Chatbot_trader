#!/usr/bin/env bash
set -euo pipefail
echo "[1/4] checando segredos (sk-, ghp_, aws_...)"
git grep -I -nE 'sk-|ghp_|AKIA|ASIA' $(git rev-list --all) && { echo ">> SEGREDO ENCONTRADO"; exit 1; } || echo "ok: sem segredos"

echo "[2/4] checando LFS"
git lfs ls-files || true

echo "[3/4] checando ignorados essenciais"
grep -qE '(^|/)venv/|(^|/)\.venv/|^analiZ/logs/' .gitignore && echo "ok: ignores básicos" || { echo ">> faltam ignores básicos"; exit 1; }

echo "[4/4] working tree clean?"
git diff --quiet && git diff --cached --quiet && echo "ok: clean" || { echo ">> há mudanças pendentes"; exit 1; }
