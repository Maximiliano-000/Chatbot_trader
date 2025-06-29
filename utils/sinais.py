# sinais.py

def interpretar_sinais_tecnicos(frases):
    pontuacao = 0
    for frase in frases:
        if "alta" in frase.lower():
            pontuacao += 1
        elif "baixa" in frase.lower():
            pontuacao -= 1
    return pontuacao
