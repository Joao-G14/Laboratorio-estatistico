"""
simulacao.py — experimentos de Monte Carlo do Módulo 3.

Sorteio é sorteio, não estatística: aqui usamos o módulo `random` da
biblioteca padrão para GERAR os experimentos. Toda medida extraída deles
(médias amostrais, desvios) continua vindo de `minhastats.py`.

Dois experimentos:
  * Lei dos Grandes Números — a frequência relativa de um evento converge
    para a probabilidade teórica conforme o número de repetições cresce;
  * Teorema Central do Limite — a distribuição das MÉDIAS de amostras
    tende à Normal conforme o tamanho da amostra cresce, mesmo quando a
    variável original é fortemente assimétrica.
"""

import random

import minhastats as ms


# ---------------------------------------------------------------------------
# Lei dos Grandes Números
# ---------------------------------------------------------------------------


def lancar_moeda(n, semente=None):
    """n lançamentos de moeda honesta: 1 = cara, 0 = coroa."""
    sorteador = random.Random(semente)
    return [sorteador.randint(0, 1) for _ in range(n)]


def lancar_dado(n, faces=6, semente=None):
    """n lançamentos de um dado honesto de `faces` lados."""
    sorteador = random.Random(semente)
    return [sorteador.randint(1, faces) for _ in range(n)]


def frequencia_relativa_acumulada(resultados, evento):
    """Frequência relativa do evento APÓS cada repetição.

        f_k = (nº de ocorrências nas k primeiras repetições) / k

    É a sequência que a Lei dos Grandes Números prevê convergir para
    P(evento). Devolvemos a série inteira, não só o valor final: o que
    ensina é justamente ver a curva oscilar muito no começo e estabilizar.
    """
    serie, ocorrencias = [], 0
    for i, resultado in enumerate(resultados, start=1):
        if resultado == evento:
            ocorrencias += 1
        serie.append(ocorrencias / i)
    return serie


def erro_da_convergencia(serie, probabilidade_teorica):
    """Distância entre a frequência simulada final e a probabilidade teórica."""
    return abs(serie[-1] - probabilidade_teorica)


# ---------------------------------------------------------------------------
# Teorema Central do Limite
# ---------------------------------------------------------------------------


def medias_amostrais(dados, tamanho_amostra, repeticoes, semente=None):
    """Sorteia `repeticoes` amostras COM reposição e devolve suas médias.

    A média de cada amostra é calculada por `minhastats.media` — é o núcleo
    do projeto que produz a distribuição amostral exibida na tela.
    """
    if tamanho_amostra < 1:
        raise ValueError("tamanho da amostra deve ser >= 1")
    if not dados:
        raise ValueError("não há dados para amostrar")
    sorteador = random.Random(semente)
    medias = []
    for _ in range(repeticoes):
        amostra = [sorteador.choice(dados) for _ in range(tamanho_amostra)]
        medias.append(ms.media(amostra))
    return medias


def erro_padrao_teorico(dados, tamanho_amostra):
    """Desvio padrão previsto para a distribuição das médias amostrais.

        σ_x̄ = σ / √n

    O TCL não diz apenas "vira Normal": diz QUAL Normal. Comparar este
    valor com o desvio observado das médias simuladas é a verificação
    quantitativa do teorema.
    """
    sigma = ms.desvio_padrao(dados, amostral=False)
    return sigma / (tamanho_amostra ** 0.5)
