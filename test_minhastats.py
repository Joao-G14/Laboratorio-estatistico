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


