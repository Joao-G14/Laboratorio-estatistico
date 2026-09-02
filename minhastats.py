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


# ---------------------------------------------------------------------------
# 4. Medidas de associação
# ---------------------------------------------------------------------------


def covariancia(x, y, amostral=True):
    """Covariância entre duas variáveis.

        cov(x, y) = Σ(xᵢ − x̄)(yᵢ − ȳ) / (n − 1)

    Sinal positivo: desvios tendem a ocorrer no mesmo sentido.
    O problema da covariância é a unidade (milha × dólar), que impede
    comparar forças de associação — daí a correlação abaixo.
    """
    _exigir_mesmo_tamanho(x, y)
    n = len(x)
    _exigir_nao_vazio(x, "x")
    if amostral and n < 2:
        raise ValueError("covariância amostral exige n >= 2")
    mx, my = media(x), media(y)
    soma = 0.0
    for i in range(n):
        soma += (x[i] - mx) * (y[i] - my)
    return soma / (n - 1 if amostral else n)


def correlacao(x, y):
    """Coeficiente de correlação linear de Pearson.

        r = cov(x, y) / (s_x · s_y)

    É a covariância padronizada pelos desvios, portanto adimensional e
    sempre em [−1, +1]. Mede SOMENTE associação LINEAR: um r ≈ 0 não
    significa "sem relação", significa "sem relação em linha reta".
    """
    _exigir_mesmo_tamanho(x, y)
    sx, sy = desvio_padrao(x), desvio_padrao(y)
    if sx == 0 or sy == 0:
        raise ValueError(
            "correlação indefinida: ao menos uma das variáveis é constante "
            "(desvio padrão zero)"
        )
    return covariancia(x, y) / (sx * sy)


def classificar_correlacao(r):
    """Traduz o valor de r em força e sentido, para o texto da interface."""
    forca = abs(r)
    if forca < 0.1:
        nome = "praticamente inexistente"
    elif forca < 0.3:
        nome = "fraca"
    elif forca < 0.5:
        nome = "moderada"
    elif forca < 0.7:
        nome = "forte"
    elif forca < 0.9:
        nome = "muito forte"
    else:
        nome = "quase perfeita"
    if forca < 0.1:
        return nome
    return f"{nome} e {'positiva' if r > 0 else 'negativa'}"


# ---------------------------------------------------------------------------
# 5. Forma da distribuição e tabelas de frequência
# ---------------------------------------------------------------------------


def assimetria(dados):
    """Coeficiente de assimetria (momento padronizado de 3ª ordem, g1).

        g1 = (1/n · Σ(xᵢ − x̄)³) / σ³        (σ populacional)

    g1 > 0 -> cauda longa à direita; g1 < 0 -> cauda à esquerda;
    g1 ≈ 0 -> aproximadamente simétrica.
    """
    n = len(dados)
    _exigir_nao_vazio(dados)
    sigma = desvio_padrao(dados, amostral=False)
    if sigma == 0:
        raise ValueError("assimetria indefinida: variável constante")
    m = media(dados)
    soma_cubos = 0.0
    for x in dados:
        soma_cubos += (x - m) ** 3
    return (soma_cubos / n) / (sigma ** 3)


def interpretar_assimetria(dados):
    """Leitura textual automática da forma, a partir de média × mediana.

    Regra adotada (declarada para não parecer arbitrária): comparamos a
    distância entre média e mediana com meio desvio padrão.
        média − mediana >  0,5·s -> assimétrica à direita
        média − mediana < −0,5·s -> assimétrica à esquerda
        caso contrário           -> aproximadamente simétrica
    """
    m, md = media(dados), mediana(dados)
    s = desvio_padrao(dados)
    diferenca = m - md
    if s == 0:
        return "Distribuição constante: todos os valores são iguais."
    if diferenca > 0.5 * s:
        return (
            f"Assimetria à DIREITA: a média ({m:.2f}) está bem acima da "
            f"mediana ({md:.2f}). Uma minoria de valores altos puxa a média "
            "para cima, então a mediana descreve melhor o caso típico."
        )
    if diferenca < -0.5 * s:
        return (
            f"Assimetria à ESQUERDA: a média ({m:.2f}) está bem abaixo da "
            f"mediana ({md:.2f}). Valores baixos extremos puxam a média."
        )
    return (
        f"Distribuição aproximadamente SIMÉTRICA: média ({m:.2f}) e mediana "
        f"({md:.2f}) estão a menos de meio desvio padrão uma da outra."
    )


def numero_classes_sturges(n):
    """Número de classes de um histograma pela regra de Sturges.

        k = 1 + 3,322 · log₁₀(n)     (arredondado para cima)
    """
    if n <= 0:
        raise ValueError("n deve ser positivo")
    if n == 1:
        return 1
    return int(math.ceil(1 + 3.322 * math.log10(n)))


def tabela_frequencias_continua(dados, k=None):
    """Tabela de frequências em classes para variáveis contínuas.

    Devolve uma lista de dicionários com, para cada classe:
        inferior, superior, ponto_medio, fi (absoluta), fri (relativa),
        Fi (acumulada), Fri (acumulada relativa), densidade.

    `densidade` = fri / largura da classe. É a altura que faz a ÁREA total do
    histograma valer 1 — a escala em que uma densidade teórica (Normal,
    Exponencial...) pode ser sobreposta aos dados, no Módulo 4.

    A última classe é fechada à direita para que o valor máximo caia dentro
    dela ([a, b) nas demais, [a, b] na última).
    """
    _exigir_nao_vazio(dados)
    n = len(dados)
    if k is None:
        k = numero_classes_sturges(n)
    minimo, maximo = min(dados), max(dados)
    if minimo == maximo:                # variável constante: uma única classe
        return [{
            "inferior": float(minimo), "superior": float(maximo),
            "ponto_medio": float(minimo), "fi": n, "fri": 1.0,
            "Fi": n, "Fri": 1.0, "densidade": 0.0,
        }]
    largura = (maximo - minimo) / k
    classes = []
    for i in range(k):
        inferior = minimo + i * largura
        # A última fronteira é fixada no próprio máximo: calculá-la como
        # minimo + k*largura acumula erro de ponto flutuante e pode deixar o
        # maior valor do conjunto de fora da tabela (bug real, pego no teste
        # test_tabela_frequencias_continua_soma_n).
        superior = float(maximo) if i == k - 1 else minimo + (i + 1) * largura
        if i == k - 1:
            contagem = sum(1 for x in dados if inferior <= x <= superior)
        else:
            contagem = sum(1 for x in dados if inferior <= x < superior)
        largura_real = superior - inferior
        classes.append({
            "inferior": inferior,
            "superior": superior,
            "ponto_medio": (inferior + superior) / 2,
            "fi": contagem,
            "fri": contagem / n,
            "densidade": (contagem / n) / largura_real if largura_real else 0.0,
        })
    acumulada = 0
    for classe in classes:
        acumulada += classe["fi"]
        classe["Fi"] = acumulada
        classe["Fri"] = acumulada / n
    return classes


def tabela_frequencias_categorica(valores):
    """Tabela de frequências para variáveis categóricas, da maior à menor."""
    _exigir_nao_vazio(valores)
    n = len(valores)
    contagem = contar_frequencias(valores)
    itens = sorted(contagem.items(), key=lambda par: (-par[1], str(par[0])))
    linhas, acumulada = [], 0
    for categoria, fi in itens:
        acumulada += fi
        linhas.append({
            "categoria": categoria, "fi": fi, "fri": fi / n,
            "Fi": acumulada, "Fri": acumulada / n,
        })
    return linhas


