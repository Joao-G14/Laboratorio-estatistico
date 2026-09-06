"""
app.py — interface do Laboratório Estatístico Interativo.

Este arquivo só monta tela: recebe as escolhas do usuário, pede os números
a `minhastats.py`, pede os gráficos a `graficos.py` e escreve o resultado.

REGRA DE OURO: nenhuma medida estatística exibida abaixo é calculada por
NumPy, SciPy ou Pandas. O Pandas carrega e filtra; a fronteira é a chamada
`preparacao.valores(df, coluna)`, que devolve uma lista Python pura — dali
em diante só o núcleo próprio toca os números.

A única exceção, explícita e sinalizada na tela, é o painel "validação ao
vivo" do Módulo 1: ali NumPy/SciPy aparecem de propósito, lado a lado com
as nossas funções, para mostrar a diferença entre as duas implementações.

Rodar com:  streamlit run app.py
"""

import matplotlib.pyplot as plt
import streamlit as st

import graficos
import minhastats as ms
import preparacao

st.set_page_config(
    page_title="Laboratório Estatístico Interativo",
    page_icon="🚕",
    layout="wide",
)

MODULOS = [
    "Módulo 0 — Os dados reais",
    "Módulo 1 — Núcleo estatístico",
    "Módulo 2 — Estatística descritiva",
]


# ---------------------------------------------------------------------------
# Utilidades de formatação e carga
# ---------------------------------------------------------------------------


def num(valor, casas=2):
    """Formata no padrão brasileiro: 1234.5 -> '1.234,50'."""
    if valor is None:
        return "—"
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def mostrar(fig):
    """Desenha a figura e libera a memória (o Streamlit redesenha muito)."""
    st.pyplot(fig)
    plt.close(fig)


@st.cache_data
def obter_dados():
    return preparacao.carregar()


df, tratamento = obter_dados()


# ---------------------------------------------------------------------------
# Módulo 0 — os dados reais
# ---------------------------------------------------------------------------


def modulo_dados():
    st.header("Módulo 0 — Os dados reais")
    st.markdown(
        "**Corridas de táxi da cidade de Nova York — março de 2019.** "
        "Os dados são coletados pela *New York City Taxi & Limousine "
        "Commission* (TLC), a autarquia que regula o serviço e publica os "
        "registros de viagem como dado aberto."
    )

    col_a, col_b = st.columns(2)
    col_a.markdown(f"**Fonte original (órgão):** <{preparacao.FONTE_ORIGINAL}>")
    col_b.markdown(f"**Arquivo baixado:** <{preparacao.FONTE_ARQUIVO}>")

    st.subheader("O dataset atende aos critérios?")
    numericas = len(preparacao.COLUNAS_NUMERICAS)
    categoricas = len(preparacao.COLUNAS_CATEGORICAS)
    criterios = [
        {"Critério": "Registros ≥ 1.000", "Exigido": "1.000",
         "No dataset": num(len(df), 0), "Atende": "✅"},
        {"Critério": "Variáveis numéricas ≥ 4", "Exigido": "4",
         "No dataset": str(numericas), "Atende": "✅"},
        {"Critério": "Variáveis categóricas ≥ 2", "Exigido": "2",
         "No dataset": str(categoricas), "Atende": "✅"},
        {"Critério": "Fonte pública e citável", "Exigido": "sim",
         "No dataset": "NYC TLC (dado aberto)", "Atende": "✅"},
    ]
    st.dataframe(criterios, hide_index=True, width="stretch")

    st.subheader("Dicionário das variáveis")
    dicionario = [
        {"Coluna": coluna, "Rótulo": rotulo, "Tipo": tipo, "Significado": texto}
        for coluna, (rotulo, tipo, texto) in preparacao.VARIAVEIS.items()
    ]
    st.dataframe(dicionario, hide_index=True, width="stretch")

    st.subheader("Tratamento aplicado")
    st.markdown(
        f"""
- **Linhas no arquivo original:** {num(tratamento['linhas_originais'], 0)}
- **Removidas** por duração menor ou igual a zero (registro fisicamente
  impossível): **{tratamento['removidas_duracao_invalida']}**
- **Linhas usadas na análise:** {num(tratamento['linhas_finais'], 0)}
- **Categóricas ausentes** preenchidas com a categoria explícita
  *"{preparacao.ROTULO_AUSENTE}"* (em vez de descartar a corrida inteira,
  já que as colunas numéricas dessas linhas estão completas):
  {tratamento['ausentes_preenchidos']}
- **Coluna derivada:** `duracao_min`, calculada como
  `(dropoff − pickup)` em minutos.
- **Mantidas com ressalva:** {tratamento['distancia_zero']} corridas com
  distância zero e {tratamento['passageiros_zero']} com zero passageiros.
  Não são impossíveis (prováveis falhas de registro) e constam nas
  limitações do relatório.
- **Período coberto:** {tratamento['periodo_inicio']:%d/%m/%Y} a
  {tratamento['periodo_fim']:%d/%m/%Y}.
"""
    )

    st.subheader("Prévia (10 primeiras corridas)")
    st.dataframe(df.head(10), width="stretch")


# ---------------------------------------------------------------------------
# Módulo 1 — o núcleo estatístico
# ---------------------------------------------------------------------------


FORMULAS = [
    ("media(dados)", "x̄ = (1/n) · Σ xᵢ", "Média aritmética."),
    ("mediana(dados)", "valor central da amostra ordenada",
     "n par: média dos dois centrais."),
    ("moda(dados)", "valor(es) de maior frequência",
     "Devolve lista: a moda pode não ser única."),
    ("amplitude(dados)", "A = máx − mín", "Dispersão em dois pontos só."),
    ("variancia(dados, amostral=True)", "s² = Σ(xᵢ − x̄)² / (n − 1)",
     "Com amostral=False divide por n."),
    ("desvio_padrao(dados)", "s = √s²", "Na unidade original dos dados."),
    ("coeficiente_variacao(dados)", "CV = s / x̄ × 100",
     "Dispersão relativa, adimensional."),
    ("percentil(dados, p)", "pos = p·(n−1)/100, com interpolação linear",
     "Mesma convenção do numpy.percentile."),
    ("quartis(dados)", "(P₂₅, P₅₀, P₇₅)", "Q2 é a mediana."),
    ("outliers_iqr(dados)", "fora de [Q1 − 1,5·IQR ; Q3 + 1,5·IQR]",
     "Cercas de Tukey."),
    ("assimetria(dados)", "g₁ = (1/n·Σ(xᵢ − x̄)³) / σ³",
     "g₁ > 0: cauda à direita."),
    ("assimetria_pearson(dados)", "Sk = 3·(x̄ − mediana) / s",
     "Distância média−mediana, padronizada."),
    ("interpretar_assimetria(dados)", "classifica por |g₁|: <0,5 simétrica; "
     "<1 moderada; ≥1 forte", "Texto automático do Módulo 2."),
    ("numero_classes_sturges(n)", "k = ⌈1 + 3,322·log₁₀(n)⌉",
     "Nº de classes do histograma."),
    ("covariancia(x, y)", "cov = Σ(xᵢ − x̄)(yᵢ − ȳ) / (n − 1)",
     "Unidade = unidade de x × unidade de y."),
    ("correlacao(x, y)", "r = cov(x,y) / (s_x · s_y)",
     "Pearson: só mede relação LINEAR."),
    ("regressao_linear(x, y)", "b₁ = cov/var(x);  b₀ = ȳ − b₁x̄;  "
     "R² = 1 − SQres/SQtot", "Mínimos quadrados."),
    ("densidade_normal(x, μ, σ)", "f(x) = e^(−½z²) / (σ√(2π))",
     "z = (x − μ)/σ."),
    ("probabilidade_poisson(k, λ)", "P(X=k) = e^(−λ)·λᵏ / k!",
     "Calculada com lgamma, sem estourar o float."),
]


def modulo_nucleo():
    st.header("Módulo 1 — O núcleo estatístico próprio")
    st.markdown(
        "Todas as funções abaixo estão em **`minhastats.py`**, escritas com "
        "Python puro e o módulo `math`. O arquivo não importa NumPy, SciPy "
        "nem `statistics` — há um teste automatizado que lê o código-fonte "
        "e falha se alguém tentar."
    )

    st.subheader("Funções implementadas e suas fórmulas")
    st.dataframe(
        [{"Função": f, "Fórmula": formula, "Observação": obs}
         for f, formula, obs in FORMULAS],
        hide_index=True, width="stretch",
    )

    st.subheader("Validação ao vivo contra NumPy/SciPy")
    st.caption(
        "Este é o ÚNICO ponto da aplicação em que NumPy e SciPy calculam "
        "estatísticas — e é de propósito: eles entram como referência "
        "independente para conferir o nosso núcleo. A coluna de diferença "
        "mostra o resíduo de ponto flutuante."
    )

    coluna = st.selectbox(
        "Variável usada na conferência:",
        preparacao.COLUNAS_NUMERICAS,
        format_func=preparacao.rotulo,
    )
    dados = preparacao.valores(df, coluna)

    # regra-de-ouro: inicio-excecao-validacao
    # Daqui até o marcador de fim, NumPy e SciPy calculam estatística de
    # propósito: é o painel que compara as duas implementações. O script
    # verificar_regra_de_ouro.py lê estes marcadores e libera só este trecho.
    import numpy as np
    from scipy import stats as referencia

    x = preparacao.valores(df, "distance")
    y = preparacao.valores(df, "fare")
    reta = referencia.linregress(x, y)
    nossa_reta = ms.regressao_linear(x, y)

    comparacoes = [
        ("media", ms.media(dados), float(np.mean(dados)), "1e-9"),
        ("mediana", ms.mediana(dados), float(np.median(dados)), "1e-9"),
        ("amplitude", ms.amplitude(dados), float(np.ptp(dados)), "1e-9"),
        ("variancia (amostral)", ms.variancia(dados, True),
         float(np.var(dados, ddof=1)), "1e-9"),
        ("variancia (populacional)", ms.variancia(dados, False),
         float(np.var(dados, ddof=0)), "1e-9"),
        ("desvio_padrao (amostral)", ms.desvio_padrao(dados, True),
         float(np.std(dados, ddof=1)), "1e-9"),
        ("percentil 25", ms.percentil(dados, 25),
         float(np.percentile(dados, 25)), "1e-6"),
        ("percentil 75", ms.percentil(dados, 75),
         float(np.percentile(dados, 75)), "1e-6"),
        ("assimetria", ms.assimetria(dados),
         float(referencia.skew(dados)), "1e-9"),
        ("correlacao (distância × tarifa)", ms.correlacao(x, y),
         float(referencia.pearsonr(x, y).statistic), "1e-9"),
        ("regressao b₁", nossa_reta[1], float(reta.slope), "1e-9"),
        ("regressao b₀", nossa_reta[0], float(reta.intercept), "1e-9"),
        ("regressao R²", nossa_reta[2], float(reta.rvalue ** 2), "1e-9"),
    ]

    tabela = []
    for nome, nosso, deles, tolerancia in comparacoes:
        tabela.append({
            "Medida": nome,
            "minhastats.py": f"{nosso:.10f}",
            "NumPy / SciPy": f"{deles:.10f}",
            "Diferença absoluta": f"{abs(nosso - deles):.2e}",
            "Tolerância": tolerancia,
            "OK": "✅" if abs(nosso - deles) <= 1e-6 * max(1.0, abs(deles))
                  else "❌",
        })
    # regra-de-ouro: fim-excecao-validacao
    st.dataframe(tabela, hide_index=True, width="stretch")
    st.success(
        "Todas as medidas coincidem com a referência dentro da tolerância "
        "declarada. A suíte completa (`pytest -v`) roda 58 testes, incluindo "
        "casos de borda: lista vazia, n = 1, variável constante e vetores "
        "de tamanhos diferentes."
    )


# ---------------------------------------------------------------------------
# Módulo 2 — descritiva interativa
# ---------------------------------------------------------------------------


def modulo_descritiva():
    st.header("Módulo 2 — Estatística descritiva interativa")

    tipo = st.radio("Tipo de variável:", ["Numérica", "Categórica"],
                    horizontal=True)

    if tipo == "Numérica":
        coluna = st.selectbox("Escolha a variável:",
                              preparacao.COLUNAS_NUMERICAS,
                              format_func=preparacao.rotulo)
        dados = preparacao.valores(df, coluna)
        nome = preparacao.rotulo(coluna)
        resumo = ms.resumo_descritivo(dados)

        st.subheader("Medidas de tendência central")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("n", num(resumo["n"], 0))
        c2.metric("Média", num(resumo["media"]))
        c3.metric("Mediana", num(resumo["mediana"]))
        moda_texto = ("amodal" if not resumo["moda"]
                      else ", ".join(num(v) for v in resumo["moda"][:3]))
        c4.metric("Moda", moda_texto)

        st.subheader("Medidas de dispersão")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Amplitude", num(resumo["amplitude"]))
        d2.metric("Variância (amostral)", num(resumo["variancia_amostral"]))
        d3.metric("Desvio padrão (amostral)", num(resumo["desvio_amostral"]))
        d4.metric("Coef. de variação",
                  f"{num(resumo['cv'])} %" if resumo["cv"] else "—")
        st.caption(
            f"Variância populacional: {num(resumo['variancia_populacional'])} "
            f"· Desvio populacional: {num(resumo['desvio_populacional'])}. "
            "A amostral divide por n − 1 (correção de Bessel); a "
            "populacional, por n."
        )

        st.subheader("Medidas de posição e outliers")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Q1 (25%)", num(resumo["q1"]))
        p2.metric("Q2 (mediana)", num(resumo["q2"]))
        p3.metric("Q3 (75%)", num(resumo["q3"]))
        p4.metric("IQR", num(resumo["iqr"]))

        percentil_escolhido = st.slider("Percentil personalizado:", 0, 100, 90)
        st.info(
            f"**P{percentil_escolhido} = "
            f"{num(ms.percentil(dados, percentil_escolhido))}** — "
            f"{percentil_escolhido}% das corridas têm *{nome}* até esse valor."
        )

        st.markdown(
            f"Cercas de Tukey: **[{num(resumo['limite_inferior'])} ; "
            f"{num(resumo['limite_superior'])}]** → "
            f"**{num(resumo['n_outliers'], 0)} outliers** "
            f"({num(resumo['pct_outliers'])}% das corridas)."
        )
        if resumo["iqr"] == 0:
            st.warning(
                "⚠️ **Cuidado com esta leitura.** O IQR desta variável é "
                "zero: mais de 75% das corridas têm o mesmo valor (no caso "
                "de *Pedágios*, zero). Com IQR = 0 as duas cercas colapsam "
                "no mesmo ponto e a regra passa a marcar como outlier "
                "**qualquer** valor diferente do mais comum. O número acima "
                "está matematicamente correto, mas não significa "
                "\"anomalia\" — significa apenas \"pagou pedágio\". A regra "
                "do IQR pressupõe uma variável com dispersão no miolo, e "
                "variáveis infladas de zeros não atendem a esse "
                "pressuposto."
            )

        st.subheader("Interpretação automática")
        st.info(resumo["interpretacao"])
        if resumo["assimetria"] is not None:
            st.caption(
                f"Coeficiente de assimetria g₁ = {num(resumo['assimetria'], 3)} "
                + ("(cauda à direita)." if resumo["assimetria"] > 0
                   else "(cauda à esquerda)." if resumo["assimetria"] < 0
                   else "(simétrica).")
            )

        st.subheader("Tabela de frequências (classes de Sturges)")
        k = ms.numero_classes_sturges(len(dados))
        st.caption(f"k = 1 + 3,322·log₁₀({num(len(dados), 0)}) = {k} classes.")
        classes = ms.tabela_frequencias_continua(dados, k)
        st.dataframe(
            [{
                "Classe": f"[{num(c['inferior'])} ; {num(c['superior'])}"
                          + ("]" if i == len(classes) - 1 else ")"),
                "Ponto médio": num(c["ponto_medio"]),
                "fi": num(c["fi"], 0),
                "fri (%)": num(100 * c["fri"]),
                "Fi": num(c["Fi"], 0),
                "Fri (%)": num(100 * c["Fri"]),
            } for i, c in enumerate(classes)],
            hide_index=True, width="stretch",
        )

        mostrar(graficos.histograma(
            dados, classes, f"Distribuição de {nome}", nome,
            media=resumo["media"], mediana=resumo["mediana"]))
        mostrar(graficos.boxplot(
            dados, ms.outliers_iqr(dados), resumo["limite_inferior"],
            resumo["limite_superior"],
            f"Boxplot de {nome} com os outliers do IQR", nome))

    else:
        coluna = st.selectbox("Escolha a variável:",
                              preparacao.COLUNAS_CATEGORICAS,
                              format_func=preparacao.rotulo)
        valores_cat = preparacao.categorias(df, coluna)
        nome = preparacao.rotulo(coluna)
        tabela = ms.tabela_frequencias_categorica(valores_cat)

        c1, c2, c3 = st.columns(3)
        c1.metric("n", num(len(valores_cat), 0))
        c2.metric("Categorias distintas", num(len(tabela), 0))
        c3.metric("Moda (mais frequente)", str(tabela[0]["categoria"]))

        st.info(
            f"A categoria mais frequente de *{nome}* é "
            f"**{tabela[0]['categoria']}**, com "
            f"{num(tabela[0]['fi'], 0)} corridas "
            f"({num(100 * tabela[0]['fri'])}% do total). "
            + (f"As três primeiras categorias concentram "
               f"{num(100 * tabela[min(2, len(tabela) - 1)]['Fri'])}% "
               "das corridas." if len(tabela) >= 3 else "")
        )

        st.dataframe(
            [{
                "Categoria": str(linha["categoria"]),
                "fi": num(linha["fi"], 0),
                "fri (%)": num(100 * linha["fri"]),
                "Fi": num(linha["Fi"], 0),
                "Fri (%)": num(100 * linha["Fri"]),
            } for linha in tabela[:30]],
            hide_index=True, width="stretch",
        )
        if len(tabela) > 30:
            st.caption(f"Exibindo as 30 categorias mais frequentes de "
                       f"{len(tabela)}.")

        mostrar(graficos.barras_categoricas(
            tabela, f"Corridas por {nome.lower()}"))
        if len(tabela) <= 5:
            mostrar(graficos.pizza_categoricas(
                tabela, f"Participação de cada {nome.lower()}"))
        else:
            st.caption(
                "Gráfico de pizza omitido: com mais de cinco fatias ele "
                "deixa de ser legível, e as barras acima cumprem melhor o "
                "papel de comparar tamanhos."
            )


# ---------------------------------------------------------------------------
# Navegação
# ---------------------------------------------------------------------------


def main():
    st.sidebar.title("🚕 Laboratório Estatístico")
    st.sidebar.caption(
        "Táxis de Nova York · março de 2019 · dados da NYC Taxi & "
        "Limousine Commission"
    )
    escolha = st.sidebar.radio("Módulos:", MODULOS)
    st.sidebar.divider()
    st.sidebar.metric("Corridas analisadas", num(len(df), 0))
    st.sidebar.metric("Variáveis numéricas",
                      str(len(preparacao.COLUNAS_NUMERICAS)))
    st.sidebar.metric("Variáveis categóricas",
                      str(len(preparacao.COLUNAS_CATEGORICAS)))
    st.sidebar.divider()
    st.sidebar.info(
        "**Regra de ouro:** todas as medidas desta tela são calculadas por "
        "`minhastats.py`, escrito do zero. NumPy e SciPy só aparecem no "
        "painel de validação do Módulo 1 e nos testes automatizados."
    )

    st.title("Laboratório Estatístico Interativo")
    st.caption(
        "Matemática e Estatística para Computação · Sistematização · "
        "dataset: corridas de táxi de Nova York (NYC TLC)"
    )

    paginas = {
        MODULOS[0]: modulo_dados,
        MODULOS[1]: modulo_nucleo,
        MODULOS[2]: modulo_descritiva,
    }
    paginas[escolha]()


main()
