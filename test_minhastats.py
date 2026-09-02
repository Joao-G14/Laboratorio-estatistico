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


