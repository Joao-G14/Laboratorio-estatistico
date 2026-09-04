"""
test_minhastats.py — validação do núcleo estatístico contra NumPy/SciPy.

Cada função de `minhastats.py` é comparada com uma implementação de
referência independente. Aqui — e SOMENTE aqui — NumPy e SciPy calculam
estatísticas: na aplicação eles nunca tocam um número exibido ao usuário.

TOLERÂNCIAS ADOTADAS (e por quê)
--------------------------------
rtol = 1e-9  para média, mediana, moda, amplitude, variância, desvio, CV,
             covariância, correlação e regressão. Somas de ponto flutuante
             feitas em ordens diferentes divergem nas últimas casas; 1e-9 é
             folgado o bastante para esse ruído e apertado o bastante para
             detectar qualquer erro real de fórmula (um n em vez de n−1, por
             exemplo, produz erro da ordem de 1/n — milhões de vezes maior).

rtol = 1e-6  para percentis e quartis. Existem nove convenções de percentil;
             usamos a interpolação linear, a mesma do método padrão do
             `numpy.percentile`, e mantemos uma folga extra porque o cálculo
             envolve multiplicação e divisão de índices.

rtol = 1e-10 para as densidades/probabilidades teóricas, que são avaliações
             diretas de exp/log — praticamente exatas.

Rodar com:  pytest -v
"""

import math
import random

import numpy as np
import pytest
from scipy import stats

import minhastats as ms

# ---------------------------------------------------------------------------
# Massas de teste
# ---------------------------------------------------------------------------

# Amostra assimétrica (gama), parecida em forma com as variáveis do dataset.
RNG = np.random.default_rng(42)
DADOS = RNG.gamma(shape=2.0, scale=9.0, size=500).tolist()

# Amostra com repetições, para exercitar moda e tabelas de frequência.
random.seed(7)
DISCRETOS = [random.randint(1, 6) for _ in range(300)]

# Par de vetores correlacionados, para covariância, Pearson e regressão.
X = RNG.uniform(0, 10, size=400).tolist()
Y = [3.5 * x + 7.0 + e for x, e in zip(X, RNG.normal(0, 2.5, size=400))]

RTOL_PADRAO = 1e-9
RTOL_PERCENTIL = 1e-6
RTOL_DISTRIBUICAO = 1e-10


# ---------------------------------------------------------------------------
# 1. Tendência central
# ---------------------------------------------------------------------------


def test_media():
    assert np.isclose(ms.media(DADOS), np.mean(DADOS), rtol=RTOL_PADRAO)


def test_media_lista_vazia_levanta_erro():
    with pytest.raises(ValueError):
        ms.media([])


def test_mediana_n_impar():
    dados = [5, 1, 9, 3, 7]
    assert np.isclose(ms.mediana(dados), np.median(dados), rtol=RTOL_PADRAO)


def test_mediana_n_par():
    dados = [5, 1, 9, 3, 7, 2]
    assert np.isclose(ms.mediana(dados), np.median(dados), rtol=RTOL_PADRAO)


def test_mediana_amostra_grande():
    assert np.isclose(ms.mediana(DADOS), np.median(DADOS), rtol=RTOL_PADRAO)


def test_moda_unimodal():
    # scipy devolve a menor das modas em caso de empate; sem empate, coincide.
    esperado = stats.mode(DISCRETOS, keepdims=False).mode
    assert esperado in ms.moda(DISCRETOS)


def test_moda_bimodal_devolve_lista():
    assert ms.moda([1, 1, 2, 2, 3]) == [1, 2]


def test_moda_amodal_devolve_lista_vazia():
    assert ms.moda([1, 2, 3, 4]) == []


# ---------------------------------------------------------------------------
# 2. Dispersão
# ---------------------------------------------------------------------------


def test_amplitude():
    assert np.isclose(ms.amplitude(DADOS),
                      np.ptp(DADOS), rtol=RTOL_PADRAO)


def test_amplitude_um_elemento_e_zero():
    assert ms.amplitude([42]) == 0


def test_variancia_amostral():
    # ddof=1 é justamente o n−1 da correção de Bessel.
    assert np.isclose(ms.variancia(DADOS, amostral=True),
                      np.var(DADOS, ddof=1), rtol=RTOL_PADRAO)


def test_variancia_populacional():
    assert np.isclose(ms.variancia(DADOS, amostral=False),
                      np.var(DADOS, ddof=0), rtol=RTOL_PADRAO)


def test_variancia_amostral_e_maior_que_populacional():
    # Consequência direta de dividir por n−1 em vez de n.
    assert ms.variancia(DADOS, True) > ms.variancia(DADOS, False)


def test_variancia_amostral_com_um_elemento_levanta_erro():
    with pytest.raises(ValueError):
        ms.variancia([3.0], amostral=True)


def test_desvio_padrao_amostral():
    assert np.isclose(ms.desvio_padrao(DADOS, amostral=True),
                      np.std(DADOS, ddof=1), rtol=RTOL_PADRAO)


def test_desvio_padrao_populacional():
    assert np.isclose(ms.desvio_padrao(DADOS, amostral=False),
                      np.std(DADOS, ddof=0), rtol=RTOL_PADRAO)


def test_coeficiente_variacao():
    esperado = 100 * np.std(DADOS, ddof=1) / abs(np.mean(DADOS))
    assert np.isclose(ms.coeficiente_variacao(DADOS), esperado,
                      rtol=RTOL_PADRAO)


def test_coeficiente_variacao_media_zero_levanta_erro():
    with pytest.raises(ValueError):
        ms.coeficiente_variacao([-2.0, 0.0, 2.0])


# ---------------------------------------------------------------------------
# 3. Posição
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0, 1, 5, 10, 25, 33.3, 50, 75, 90, 99, 100])
def test_percentil_bate_com_numpy(p):
    assert np.isclose(ms.percentil(DADOS, p), np.percentile(DADOS, p),
                      rtol=RTOL_PERCENTIL)


def test_percentil_fora_do_intervalo_levanta_erro():
    with pytest.raises(ValueError):
        ms.percentil(DADOS, 101)


def test_quartis():
    q1, q2, q3 = ms.quartis(DADOS)
    esperados = np.percentile(DADOS, [25, 50, 75])
    assert np.allclose([q1, q2, q3], esperados, rtol=RTOL_PERCENTIL)


def test_q2_e_a_mediana():
    _, q2, _ = ms.quartis(DADOS)
    assert np.isclose(q2, ms.mediana(DADOS), rtol=RTOL_PADRAO)


def test_intervalo_interquartil():
    q1, q3 = np.percentile(DADOS, [25, 75])
    assert np.isclose(ms.intervalo_interquartil(DADOS), q3 - q1,
                      rtol=RTOL_PERCENTIL)


def test_outliers_iqr_encontra_o_extremo_plantado():
    base = list(DISCRETOS) + [999.0]
    assert 999.0 in ms.outliers_iqr(base)


def test_outliers_iqr_respeita_as_cercas_de_tukey():
    inferior, superior = ms.limites_outliers(DADOS)
    esperados = [x for x in DADOS if x < inferior or x > superior]
    assert ms.outliers_iqr(DADOS) == esperados


# ---------------------------------------------------------------------------
# 4. Associação
# ---------------------------------------------------------------------------


def test_covariancia_amostral():
    # np.cov devolve a matriz; o termo [0][1] é a covariância entre x e y.
    assert np.isclose(ms.covariancia(X, Y), np.cov(X, Y, ddof=1)[0][1],
                      rtol=RTOL_PADRAO)


def test_covariancia_populacional():
    assert np.isclose(ms.covariancia(X, Y, amostral=False),
                      np.cov(X, Y, ddof=0)[0][1], rtol=RTOL_PADRAO)


def test_covariancia_tamanhos_diferentes_levanta_erro():
    with pytest.raises(ValueError):
        ms.covariancia([1, 2, 3], [1, 2])


def test_correlacao_pearson():
    esperado = stats.pearsonr(X, Y).statistic
    assert np.isclose(ms.correlacao(X, Y), esperado, rtol=RTOL_PADRAO)


def test_correlacao_bate_com_numpy_corrcoef():
    assert np.isclose(ms.correlacao(X, Y), np.corrcoef(X, Y)[0][1],
                      rtol=RTOL_PADRAO)


def test_correlacao_de_variavel_consigo_mesma_e_um():
    assert np.isclose(ms.correlacao(X, X), 1.0, rtol=RTOL_PADRAO)


def test_correlacao_com_variavel_constante_levanta_erro():
    with pytest.raises(ValueError):
        ms.correlacao([1, 2, 3, 4], [5, 5, 5, 5])


# ---------------------------------------------------------------------------
# 5. Forma e tabelas de frequência
# ---------------------------------------------------------------------------


def test_assimetria_bate_com_scipy():
    assert np.isclose(ms.assimetria(DADOS), stats.skew(DADOS),
                      rtol=RTOL_PADRAO)


def test_assimetria_positiva_em_dados_com_cauda_a_direita():
    assert ms.assimetria(DADOS) > 0


def test_numero_classes_sturges():
    n = len(DADOS)
    assert ms.numero_classes_sturges(n) == math.ceil(1 + 3.322 * math.log10(n))


def test_tabela_frequencias_continua_soma_n():
    tabela = ms.tabela_frequencias_continua(DADOS)
    assert sum(linha["fi"] for linha in tabela) == len(DADOS)


def test_tabela_frequencias_continua_frequencia_relativa_soma_um():
    tabela = ms.tabela_frequencias_continua(DADOS)
    assert np.isclose(sum(linha["fri"] for linha in tabela), 1.0,
                      rtol=RTOL_PADRAO)


def test_tabela_frequencias_densidade_tem_area_um():
    """A soma de (densidade × largura da classe) precisa valer 1.

    É essa propriedade que permite sobrepor uma densidade teórica ao
    histograma no Módulo 4 sem que as escalas briguem.
    """
    tabela = ms.tabela_frequencias_continua(DADOS)
    area = sum(linha["densidade"] * (linha["superior"] - linha["inferior"])
               for linha in tabela)
    assert np.isclose(area, 1.0, rtol=RTOL_PADRAO)


def test_tabela_frequencias_continua_bate_com_histograma_do_numpy():
    k = ms.numero_classes_sturges(len(DADOS))
    contagens_numpy, _ = np.histogram(DADOS, bins=k)
    tabela = ms.tabela_frequencias_continua(DADOS, k)
    assert [linha["fi"] for linha in tabela] == list(contagens_numpy)


def test_tabela_frequencias_categorica():
    valores = ["a", "b", "a", "c", "a", "b"]
    tabela = ms.tabela_frequencias_categorica(valores)
    assert tabela[0]["categoria"] == "a" and tabela[0]["fi"] == 3
    assert np.isclose(tabela[-1]["Fri"], 1.0, rtol=RTOL_PADRAO)


# ---------------------------------------------------------------------------
# 6. Regressão linear
# ---------------------------------------------------------------------------


def test_regressao_coeficientes_batem_com_scipy():
    b0, b1, _ = ms.regressao_linear(X, Y)
    referencia = stats.linregress(X, Y)
    assert np.isclose(b1, referencia.slope, rtol=RTOL_PADRAO)
    assert np.isclose(b0, referencia.intercept, rtol=RTOL_PADRAO)


def test_regressao_r2_bate_com_scipy():
    _, _, r2 = ms.regressao_linear(X, Y)
    referencia = stats.linregress(X, Y)
    assert np.isclose(r2, referencia.rvalue ** 2, rtol=RTOL_PADRAO)


def test_regressao_coeficientes_batem_com_polyfit():
    b0, b1, _ = ms.regressao_linear(X, Y)
    coef_numpy = np.polyfit(X, Y, 1)
    assert np.allclose([b1, b0], coef_numpy, rtol=1e-8)


def test_r2_e_o_quadrado_da_correlacao_na_regressao_simples():
    # Identidade da regressão simples: R² = r². Bom teste de consistência.
    _, _, r2 = ms.regressao_linear(X, Y)
    r = ms.correlacao(X, Y)
    assert np.isclose(r2, r ** 2, rtol=RTOL_PADRAO)


def test_prever_reproduz_a_reta():
    b0, b1, _ = ms.regressao_linear(X, Y)
    assert np.isclose(ms.prever(b0, b1, 5.0), b0 + b1 * 5.0, rtol=RTOL_PADRAO)


def test_regressao_com_x_constante_levanta_erro():
    with pytest.raises(ValueError):
        ms.regressao_linear([2, 2, 2, 2], [1, 2, 3, 4])


def test_erro_padrao_estimativa_bate_com_residuos_do_numpy():
    b0, b1, _ = ms.regressao_linear(X, Y)
    residuos = np.array(Y) - (b0 + b1 * np.array(X))
    esperado = np.sqrt(np.sum(residuos ** 2) / (len(X) - 2))
    assert np.isclose(ms.erro_padrao_estimativa(X, Y), esperado,
                      rtol=RTOL_PADRAO)


# ---------------------------------------------------------------------------
# 7. Distribuições teóricas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x", [-3.0, -0.5, 0.0, 1.25, 4.0, 9.9])
def test_densidade_normal(x):
    assert np.isclose(ms.densidade_normal(x, 2.0, 1.7),
                      stats.norm.pdf(x, loc=2.0, scale=1.7),
                      rtol=RTOL_DISTRIBUICAO)


@pytest.mark.parametrize("x", [0.0, 0.5, 2.0, 7.5])
def test_densidade_exponencial(x):
    lam = 0.4
    assert np.isclose(ms.densidade_exponencial(x, lam),
                      stats.expon.pdf(x, scale=1 / lam),
                      rtol=RTOL_DISTRIBUICAO)


@pytest.mark.parametrize("x", [-1.0, 0.0, 2.5, 5.0, 6.0])
def test_densidade_uniforme(x):
    a, b = 0.0, 5.0
    assert np.isclose(ms.densidade_uniforme(x, a, b),
                      stats.uniform.pdf(x, loc=a, scale=b - a),
                      rtol=RTOL_DISTRIBUICAO)


@pytest.mark.parametrize("k", [0, 1, 2, 3, 5, 10])
def test_probabilidade_poisson(k):
    lam = 1.54
    assert np.isclose(ms.probabilidade_poisson(k, lam),
                      stats.poisson.pmf(k, lam), rtol=RTOL_DISTRIBUICAO)


def test_poisson_soma_das_probabilidades_tende_a_um():
    lam = 1.54
    total = sum(ms.probabilidade_poisson(k, lam) for k in range(0, 40))
    assert np.isclose(total, 1.0, rtol=1e-9)


@pytest.mark.parametrize("k", [0, 1, 4, 7, 10])
def test_probabilidade_binomial(k):
    n, p = 10, 0.37
    assert np.isclose(ms.probabilidade_binomial(k, n, p),
                      stats.binom.pmf(k, n, p), rtol=RTOL_DISTRIBUICAO)


def test_binomial_soma_das_probabilidades_e_um():
    n, p = 10, 0.37
    total = sum(ms.probabilidade_binomial(k, n, p) for k in range(n + 1))
    assert np.isclose(total, 1.0, rtol=1e-12)


def test_binomial_com_p_zero():
    assert ms.probabilidade_binomial(0, 5, 0.0) == 1.0
    assert ms.probabilidade_binomial(1, 5, 0.0) == 0.0


# ---------------------------------------------------------------------------
# 8. Resumo descritivo e regra de ouro
# ---------------------------------------------------------------------------


def test_resumo_descritivo_traz_todas_as_chaves():
    resumo = ms.resumo_descritivo(DADOS)
    esperadas = {"n", "media", "mediana", "moda", "minimo", "maximo",
                 "amplitude", "variancia_amostral", "variancia_populacional",
                 "desvio_amostral", "desvio_populacional", "q1", "q2", "q3",
                 "iqr", "limite_inferior", "limite_superior", "n_outliers",
                 "pct_outliers", "cv", "assimetria", "interpretacao"}
    assert esperadas.issubset(resumo.keys())


def test_resumo_descritivo_concorda_com_as_funcoes_isoladas():
    resumo = ms.resumo_descritivo(DADOS)
    assert np.isclose(resumo["media"], np.mean(DADOS), rtol=RTOL_PADRAO)
    assert np.isclose(resumo["desvio_amostral"], np.std(DADOS, ddof=1),
                      rtol=RTOL_PADRAO)


def test_regra_de_ouro_o_nucleo_nao_importa_biblioteca_estatistica():
    """O núcleo não pode depender de NumPy, SciPy, statistics ou pandas.

    Este teste lê o próprio código-fonte de minhastats.py. Se alguém um dia
    "resolver" um bug importando np.mean, o teste quebra — que é exatamente
    o objetivo do trabalho.
    """
    with open("minhastats.py", encoding="utf-8") as arquivo:
        codigo = arquivo.read()
    for proibido in ("import numpy", "import scipy", "import statistics",
                     "import pandas", "from numpy", "from scipy",
                     "from statistics", "from pandas"):
        assert proibido not in codigo, f"minhastats.py não pode conter '{proibido}'"
