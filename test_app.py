"""
test_app.py — teste de fumaça da interface.

`pytest` sobe a aplicação inteira em memória (sem navegador) com o
`AppTest` do próprio Streamlit, navega pelos sete módulos e verifica que
nenhum deles levanta exceção. É o teste que pega o erro clássico de véspera
de entrega: um módulo que quebra porque alguém renomeou uma coluna.

Rodar com:  pytest test_app.py -v
"""

import pytest
from streamlit.testing.v1 import AppTest

from app import MODULOS

TEMPO_LIMITE = 180          # segundos; o Módulo 6 monta a matriz inteira


def abrir(modulo):
    """Sobe a aplicação e navega até o módulo pedido."""
    app = AppTest.from_file("app.py", default_timeout=TEMPO_LIMITE)
    app.run()
    assert not app.exception, f"a aplicação quebrou ao iniciar: {app.exception}"
    app.sidebar.radio[0].set_value(modulo).run()
    return app


@pytest.mark.parametrize("modulo", MODULOS)
def test_modulo_roda_sem_excecao(modulo):
    app = abrir(modulo)
    assert not app.exception, f"{modulo} levantou: {app.exception}"


def test_titulo_da_aplicacao():
    app = AppTest.from_file("app.py", default_timeout=TEMPO_LIMITE)
    app.run()
    assert "Laboratório Estatístico Interativo" in app.title[0].value


def test_modulo_5_responde_a_predicao():
    """Mexer no campo de predição tem de recalcular sem quebrar."""
    app = abrir(MODULOS[5])
    campo = app.number_input[0]
    app = campo.set_value(10.0).run()
    assert not app.exception
    # Com x = 10 milhas a reta prevê um valor positivo e plausível de tarifa.
    textos = " ".join(elemento.value for elemento in app.success)
    assert "Tarifa" in textos


def test_modulo_3_aceita_mudanca_de_parametros():
    """Os controles da simulação (sliders) devem reexecutar limpo."""
    app = abrir(MODULOS[3])
    app = app.slider[0].set_value(1000).run()
    assert not app.exception
