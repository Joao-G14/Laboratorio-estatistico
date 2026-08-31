"""
minhastats.py — Núcleo estatístico do Laboratório Estatístico Interativo.

REGRA DE OURO DO PROJETO
------------------------
Este arquivo não importa NumPy, SciPy nem `statistics`. Tudo o que é exibido
ao usuário na aplicação é calculado aqui, com Python puro e o módulo `math`
(apenas para sqrt, exp, log, pi e lgamma — funções matemáticas elementares,
não funções estatísticas prontas).

NumPy/SciPy aparecem somente em `test_minhastats.py`, como referência
independente para validar estas implementações.

Convenções adotadas em todo o arquivo:
  * `dados` é uma sequência de números (list, tuple...);
  * funções que dependem de outras reaproveitam as daqui (composição),
    para que uma correção de fórmula valha para todo o módulo;
  * casos degenerados (lista vazia, n = 1, desvio zero) levantam ValueError
    com mensagem explicativa, em vez de devolver um número sem sentido.
"""

import math

# ---------------------------------------------------------------------------
# Validações internas
# ---------------------------------------------------------------------------


def _exigir_nao_vazio(dados, nome="dados"):
    """Garante que a sequência tem ao menos um elemento."""
    if len(dados) == 0:
        raise ValueError(f"{nome}: sequência vazia — medida indefinida")


def _exigir_mesmo_tamanho(x, y):
    """Garante que dois vetores são pareáveis ponto a ponto."""
    if len(x) != len(y):
        raise ValueError(
            f"x e y precisam ter o mesmo tamanho (recebi {len(x)} e {len(y)})"
        )


# ---------------------------------------------------------------------------
# 1. Medidas de tendência central
# ---------------------------------------------------------------------------


def media(dados):
    """Média aritmética.

        x̄ = (1/n) · Σ xᵢ

    É o "centro de massa" da distribuição: o ponto de equilíbrio da régua.
    Por isso é sensível a valores extremos — um único outlier muito alto
    desloca a média, mas quase não move a mediana.
    """
    _exigir_nao_vazio(dados)
    soma = 0.0
    for valor in dados:
        soma += valor
    return soma / len(dados)


def mediana(dados):
    """Mediana: o valor que divide a amostra ordenada em duas metades.

        n ímpar -> elemento da posição central
        n par   -> média dos dois elementos centrais

    Robusta a outliers: depende da posição dos valores, não da magnitude.
    """
    _exigir_nao_vazio(dados)
    ordenados = sorted(dados)
    n = len(ordenados)
    meio = n // 2
    if n % 2 == 1:                      # n ímpar: existe um elemento central
        return float(ordenados[meio])
    # n par: média dos dois centrais (índices meio-1 e meio)
    return (ordenados[meio - 1] + ordenados[meio]) / 2.0


def contar_frequencias(valores):
    """Dicionário {valor: quantas vezes aparece}. Base da moda e das tabelas."""
    contagem = {}
    for v in valores:
        contagem[v] = contagem.get(v, 0) + 1
    return contagem


def moda(dados):
    """Moda: o(s) valor(es) de maior frequência.

    Devolve sempre uma LISTA, porque a moda pode não ser única:
    [7] é unimodal, [3, 9] é bimodal. Se todos os valores aparecem
    o mesmo número de vezes, a distribuição é amodal e a lista vem vazia.
    """
    _exigir_nao_vazio(dados)
    contagem = contar_frequencias(dados)
    maior = max(contagem.values())
    if maior == 1 and len(contagem) == len(dados):
        return []                       # nenhum valor se repete: amodal
    return sorted([v for v, f in contagem.items() if f == maior])


# ---------------------------------------------------------------------------
# 2. Medidas de dispersão
# ---------------------------------------------------------------------------


def amplitude(dados):
    """Amplitude total: máximo − mínimo.

    Usa apenas dois pontos, então é a medida de dispersão mais frágil:
    um outlier isolado já dita o valor inteiro.
    """
    _exigir_nao_vazio(dados)
    return float(max(dados) - min(dados))


def variancia(dados, amostral=True):
    """Variância — média dos desvios quadráticos em relação à média.

        amostral=True   s² = Σ(xᵢ − x̄)² / (n − 1)      (correção de Bessel)
        amostral=False  σ² = Σ(xᵢ − x̄)² / n

    Por que n − 1? Porque x̄ foi estimada a partir dos mesmos dados: os
    desvios em torno dela já são, por construção, os menores possíveis.
    Dividir por n subestimaria sistematicamente a dispersão da população.
    O n − 1 devolve esse grau de liberdade "gasto" na estimativa da média.
    """
    n = len(dados)
    _exigir_nao_vazio(dados)
    if amostral and n < 2:
        raise ValueError("variância amostral exige n >= 2")
    m = media(dados)
    soma_quadrados = 0.0
    for x in dados:
        soma_quadrados += (x - m) ** 2
    return soma_quadrados / (n - 1 if amostral else n)


def desvio_padrao(dados, amostral=True):
    """Raiz quadrada da variância — dispersão na MESMA unidade dos dados.

        s = √s²

    É por isso que se reporta o desvio e não a variância: "R$ 11,55" é
    legível; "133,4 reais²" não é.
    """
    return math.sqrt(variancia(dados, amostral))


def coeficiente_variacao(dados, amostral=True, em_percentual=True):
    """Coeficiente de variação: dispersão relativa ao tamanho da média.

        CV = s / x̄          (× 100 para ler em %)

    Adimensional, o que permite comparar a dispersão de variáveis em
    escalas diferentes (distância em milhas × valor em dólares).
    Referência usual: CV < 15% baixa, 15–30% média, > 30% alta dispersão.
    """
    m = media(dados)
    if m == 0:
        raise ValueError("CV indefinido: a média é zero")
    cv = desvio_padrao(dados, amostral) / abs(m)
    return cv * 100 if em_percentual else cv


# ---------------------------------------------------------------------------
# 3. Medidas de posição
# ---------------------------------------------------------------------------


def percentil(dados, p):
    """Percentil p (0 a 100) por interpolação linear entre vizinhos.

        posição = p · (n − 1) / 100        (índice real no vetor ordenado)
        valor   = x[k] + f · (x[k+1] − x[k])

    onde k é a parte inteira da posição e f a parte fracionária.

    Existem pelo menos nove convenções de percentil na literatura; esta é
    a mesma usada por `numpy.percentile` com o método padrão ("linear"),
    escolhida justamente para que a validação seja direta e honesta.
    """
    _exigir_nao_vazio(dados)
    if not 0 <= p <= 100:
        raise ValueError("p deve estar entre 0 e 100")
    ordenados = sorted(dados)
    n = len(ordenados)
    if n == 1:
        return float(ordenados[0])
    posicao = p * (n - 1) / 100.0
    k = int(posicao)                    # parte inteira -> índice de baixo
    fracao = posicao - k                # parte fracionária -> peso
    if k + 1 >= n:                      # p = 100 cai exatamente no último
        return float(ordenados[n - 1])
    return ordenados[k] + fracao * (ordenados[k + 1] - ordenados[k])


def quartis(dados):
    """(Q1, Q2, Q3) — os percentis 25, 50 e 75.

    Q2 é a mediana. Entre Q1 e Q3 está a "metade do meio" dos dados.
    """
    return percentil(dados, 25), percentil(dados, 50), percentil(dados, 75)


def intervalo_interquartil(dados):
    """IQR = Q3 − Q1: a amplitude da metade central, imune aos extremos."""
    q1, _, q3 = quartis(dados)
    return q3 - q1


def limites_outliers(dados, fator=1.5):
    """Cercas de Tukey para detecção de outliers.

        limite inferior = Q1 − 1,5 · IQR
        limite superior = Q3 + 1,5 · IQR

    O fator 1,5 é convenção (com 3,0 fala-se em outlier extremo).
    """
    q1, _, q3 = quartis(dados)
    iqr = q3 - q1
    return q1 - fator * iqr, q3 + fator * iqr


def outliers_iqr(dados, fator=1.5):
    """Lista dos valores fora das cercas de Tukey."""
    inferior, superior = limites_outliers(dados, fator)
    return [x for x in dados if x < inferior or x > superior]


