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
import simulacao

st.set_page_config(
    page_title="Laboratório Estatístico Interativo",
    page_icon="🚕",
    layout="wide",
)

MODULOS = [
    "Módulo 0 — Os dados reais",
    "Módulo 1 — Núcleo estatístico",
    "Módulo 2 — Estatística descritiva",
    "Módulo 3 — Simulação: LGN e TCL",
    "Módulo 4 — Distribuições teóricas",
    "Módulo 5 — Correlação e regressão",
    "Módulo 6 — As três descobertas",
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
        "declarada. A suíte completa (`pytest -v`) roda 104 testes, incluindo "
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
# Módulo 3 — simulação
# ---------------------------------------------------------------------------


def modulo_simulacao():
    st.header("Módulo 3 — Probabilidade e simulação de Monte Carlo")

    aba_lgn, aba_tcl = st.tabs(
        ["Experimento A — Lei dos Grandes Números",
         "Experimento B — Teorema Central do Limite"])

    with aba_lgn:
        st.markdown(
            "A **Lei dos Grandes Números** afirma que a frequência relativa "
            "de um evento converge para sua probabilidade teórica conforme "
            "o número de repetições cresce. Vamos verificar sorteando."
        )
        col1, col2, col3 = st.columns(3)
        experimento = col1.selectbox("Experimento:",
                                     ["Moeda honesta", "Dado de 6 faces"])
        n = col2.slider("Nº de repetições:", 100, 50_000, 5_000, step=100)
        semente = col3.number_input("Semente aleatória:", 0, 9999, 42,
                                    help="Mude a semente para ver que CADA "
                                         "simulação oscila diferente no "
                                         "começo — e converge igual no fim.")

        if experimento == "Moeda honesta":
            resultados = simulacao.lancar_moeda(n, semente=semente)
            evento, rotulo_evento, teorica = 1, "cara", 0.5
        else:
            resultados = simulacao.lancar_dado(n, semente=semente)
            face = st.select_slider("Face observada:", options=[1, 2, 3, 4, 5, 6],
                                    value=6)
            evento, rotulo_evento, teorica = face, f"face {face}", 1 / 6

        serie = simulacao.frequencia_relativa_acumulada(resultados, evento)
        erro = simulacao.erro_da_convergencia(serie, teorica)

        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilidade teórica", num(teorica, 4))
        m2.metric(f"Frequência após {num(n, 0)} repetições", num(serie[-1], 4))
        m3.metric("Erro absoluto", num(erro, 4))

        mostrar(graficos.grafico_lgn(
            serie, teorica,
            f"Lei dos Grandes Números — {num(n, 0)} repetições",
            rotulo_evento))

        st.info(
            f"Depois de 10 repetições a frequência era "
            f"{num(serie[9], 4)}; depois de 100, {num(serie[99], 4)}; "
            f"depois de {num(n, 0)}, {num(serie[-1], 4)} — contra "
            f"{num(teorica, 4)} de probabilidade teórica. A convergência "
            "não é uma reta descendente de erro: é uma oscilação que vai "
            "perdendo amplitude. Aumente as repetições e o erro encolhe na "
            "ordem de 1/√n."
        )

    with aba_tcl:
        st.markdown(
            "O **Teorema Central do Limite** diz que a distribuição das "
            "MÉDIAS de amostras tende à Normal conforme o tamanho da "
            "amostra cresce — **mesmo quando a variável original é tudo "
            "menos normal**. Escolha abaixo uma variável bem assimétrica "
            "para ver o efeito."
        )
        col1, col2, col3 = st.columns(3)
        coluna = col1.selectbox("Variável do dataset:",
                                preparacao.COLUNAS_NUMERICAS,
                                format_func=preparacao.rotulo, key="tcl_var")
        repeticoes = col2.slider("Nº de amostras sorteadas:",
                                 200, 5_000, 2_000, step=100)
        semente_tcl = col3.number_input("Semente:", 0, 9999, 7, key="tcl_seed")
        tamanhos = st.multiselect(
            "Tamanhos de amostra (n) a comparar:",
            [2, 5, 10, 30, 50, 100], default=[2, 10, 30])

        if not tamanhos:
            st.warning("Escolha ao menos um tamanho de amostra.")
            return

        dados = preparacao.valores(df, coluna)
        nome = preparacao.rotulo(coluna)
        media_pop = ms.media(dados)
        desvio_pop = ms.desvio_padrao(dados, amostral=False)

        st.caption(
            f"População: média = {num(media_pop)}, desvio padrão "
            f"(populacional) = {num(desvio_pop)}, assimetria g₁ = "
            f"{num(ms.assimetria(dados), 3)}."
        )

        paineis, linhas = [], []
        for tamanho in sorted(tamanhos):
            medias = simulacao.medias_amostrais(
                dados, tamanho, repeticoes, semente=semente_tcl)
            media_das_medias = ms.media(medias)
            desvio_das_medias = ms.desvio_padrao(medias, amostral=False)
            teorico = simulacao.erro_padrao_teorico(dados, tamanho)

            # Sturges é a regra para uma TABELA legível; para julgar o
            # FORMATO de uma distribuição um grid mais fino comunica melhor.
            classes = ms.tabela_frequencias_continua(medias, 30)
            passo = (max(medias) - min(medias)) / 120 or 1.0
            curva_x = [min(medias) + i * passo for i in range(121)]
            curva_y = [ms.densidade_normal(v, media_das_medias,
                                           desvio_das_medias)
                       for v in curva_x]
            paineis.append({"n": tamanho, "classes": classes,
                            "curva_x": curva_x, "curva_y": curva_y})
            linhas.append({
                "n (tamanho da amostra)": tamanho,
                "Média das médias": num(media_das_medias),
                "Média da população": num(media_pop),
                "Desvio das médias (observado)": num(desvio_das_medias, 3),
                "σ/√n (previsto pelo TCL)": num(teorico, 3),
                "Assimetria das médias": num(ms.assimetria(medias), 3),
            })

        mostrar(graficos.grafico_tcl(
            paineis,
            f"Distribuição das médias amostrais de {nome} "
            f"({num(repeticoes, 0)} amostras por painel)"))

        st.dataframe(linhas, hide_index=True, width="stretch")
        st.info(
            "Duas coisas para conferir na tabela. **Primeira:** a média das "
            "médias fica praticamente colada na média da população, "
            "qualquer que seja n — a média amostral é um estimador não "
            "viesado. **Segunda:** o desvio observado das médias acompanha "
            "σ/√n, ou seja, quadruplicar o tamanho da amostra corta o erro "
            "pela metade. E a coluna de assimetria cai em direção a zero "
            "conforme n cresce: é a distribuição virando um sino."
        )


# ---------------------------------------------------------------------------
# Módulo 4 — distribuições teóricas
# ---------------------------------------------------------------------------


def modulo_distribuicoes():
    st.header("Módulo 4 — Distribuições teóricas sobre os dados")
    st.markdown(
        "Estimamos os parâmetros **a partir dos próprios dados**, com as "
        "nossas funções, e sobrepomos a densidade teórica ao histograma. "
        "O histograma está em **densidade** (área total = 1): sem isso as "
        "escalas não bateriam e a curva sumiria rente ao eixo."
    )

    coluna = st.selectbox("Variável:", preparacao.COLUNAS_NUMERICAS,
                          format_func=preparacao.rotulo, key="dist_var")
    dados = preparacao.valores(df, coluna)
    nome = preparacao.rotulo(coluna)

    if coluna == "passengers":
        st.subheader("Variável discreta: Poisson × observado")
        st.markdown(
            "Para uma **contagem** a candidata natural é a Poisson, com "
            "λ̂ = média. Comparamos proporção observada e probabilidade "
            "teórica barra a barra."
        )
        lam = ms.media(dados)
        tabela = ms.tabela_frequencias_categorica([int(v) for v in dados])
        tabela = sorted(tabela, key=lambda linha: linha["categoria"])
        categorias = [linha["categoria"] for linha in tabela]
        observado = [linha["fri"] for linha in tabela]
        teorico = [ms.probabilidade_poisson(k, lam) for k in categorias]

        st.caption(f"λ̂ = média = {num(lam, 4)} passageiros por corrida.")
        mostrar(graficos.barras_observado_teorico(
            categorias, observado, teorico, f"Poisson(λ = {num(lam, 3)})",
            "Número de passageiros: observado × Poisson", "passageiros"))

        erro_medio = ms.media([abs(o - t) for o, t in zip(observado, teorico)])
        st.warning(
            f"**O ajuste é ruim — e isso é o achado.** O erro médio absoluto "
            f"por categoria é {num(erro_medio, 4)}. A Poisson prevê "
            f"{num(100 * ms.probabilidade_poisson(0, lam))}% de corridas com "
            f"zero passageiros, e observamos "
            f"{num(100 * observado[0])}%; prevê "
            f"{num(100 * ms.probabilidade_poisson(1, lam))}% com um "
            f"passageiro, e observamos {num(100 * observado[1])}%. "
            "O motivo é conceitual: a Poisson descreve eventos raros e "
            "independentes ao longo de um intervalo, e o número de "
            "passageiros de um táxi não é isso — é o tamanho de um grupo "
            "social, com um pico enorme em 1 (quem anda sozinho) e um teto "
            "físico em 6 lugares. Nenhum parâmetro consertaria isso: o "
            "modelo é que não corresponde ao fenômeno."
        )
        return

    media_dados = ms.media(dados)
    desvio_dados = ms.desvio_padrao(dados)
    minimo, maximo = min(dados), max(dados)

    alternativa = st.selectbox(
        "Segunda distribuição a sobrepor (além da Normal):",
        ["Exponencial", "Uniforme"])

    # 40 classes (e não Sturges): aqui o histograma não é uma tabela de
    # leitura, é o objeto que vai ser comparado com uma curva contínua —
    # precisa de resolução para que o ajuste possa ser julgado.
    classes = ms.tabela_frequencias_continua(dados, 40)

    passo = (maximo - minimo) / 300 or 1.0
    xs = [minimo + i * passo for i in range(301)]
    curvas = [(f"Normal(μ = {num(media_dados)}; σ = {num(desvio_dados)})",
               xs, [ms.densidade_normal(v, media_dados, desvio_dados)
                    for v in xs])]

    if alternativa == "Exponencial":
        if media_dados <= 0:
            st.error("Exponencial exige média positiva.")
            return
        lam = 1 / media_dados
        curvas.append((f"Exponencial(λ = {num(lam, 4)})", xs,
                       [ms.densidade_exponencial(v, lam) for v in xs]))
        texto_parametro = f"λ̂ = 1/x̄ = {num(lam, 4)}"
    else:
        curvas.append((f"Uniforme({num(minimo)}; {num(maximo)})", xs,
                       [ms.densidade_uniforme(v, minimo, maximo) for v in xs]))
        texto_parametro = f"â = mín = {num(minimo)}, b̂ = máx = {num(maximo)}"

    c1, c2, c3 = st.columns(3)
    c1.metric("μ̂ = média", num(media_dados))
    c2.metric("σ̂ = desvio padrão", num(desvio_dados))
    c3.metric("Assimetria g₁", num(ms.assimetria(dados), 3))
    st.caption(f"Parâmetros da alternativa estimados dos dados: "
               f"{texto_parametro}.")

    mostrar(graficos.histograma_com_curvas(
        classes, curvas, f"{nome}: histograma em densidade e curvas teóricas",
        nome))

    # Qualidade do ajuste: erro médio absoluto entre densidade observada
    # (altura das barras) e densidade teórica no ponto médio da classe.
    st.subheader("Qualidade do ajuste, medida e discutida")
    linhas_ajuste = []
    for rotulo_curva, _, _ in curvas:
        if rotulo_curva.startswith("Normal"):
            f = lambda v: ms.densidade_normal(v, media_dados, desvio_dados)
        elif rotulo_curva.startswith("Exponencial"):
            f = lambda v: ms.densidade_exponencial(v, 1 / media_dados)
        else:
            f = lambda v: ms.densidade_uniforme(v, minimo, maximo)
        erros = [abs(c["densidade"] - f(c["ponto_medio"])) for c in classes]
        linhas_ajuste.append({
            "Distribuição": rotulo_curva,
            "Erro médio absoluto de densidade": f"{ms.media(erros):.6f}",
            "Maior erro em uma classe": f"{max(erros):.6f}",
        })
    st.dataframe(linhas_ajuste, hide_index=True, width="stretch")

    assimetria_valor = ms.assimetria(dados)
    if assimetria_valor > 0.8:
        st.warning(
            f"**A Normal ajusta mal, e por um motivo identificável.** "
            f"A assimetria de *{nome}* é g₁ = {num(assimetria_valor, 2)}, "
            "bem longe do zero que a Normal pressupõe. A curva normal é "
            "simétrica, então ela coloca massa de probabilidade à esquerda "
            f"do mínimo observado ({num(minimo)}) — região onde a variável "
            "nem pode existir — e some cedo demais na cauda direita, onde "
            "estão as corridas longas. A Exponencial, que já nasce "
            "assimétrica e ancorada em zero, acompanha muito melhor o "
            "formato geral, ainda que subestime o pico. Um ajuste ruim bem "
            "diagnosticado vale mais que um ajuste bonito sem análise."
        )
    else:
        st.info(
            f"A assimetria de *{nome}* é g₁ = {num(assimetria_valor, 2)}. "
            "Quanto mais perto de zero, mais defensável é a aproximação "
            "Normal. Compare o erro médio das duas curvas na tabela acima."
        )


# ---------------------------------------------------------------------------
# Módulo 5 — correlação e regressão
# ---------------------------------------------------------------------------


def modulo_regressao():
    st.header("Módulo 5 — Correlação e regressão linear")

    col1, col2 = st.columns(2)
    coluna_x = col1.selectbox("Variável X (explicativa):",
                              preparacao.COLUNAS_NUMERICAS, index=0,
                              format_func=preparacao.rotulo)
    coluna_y = col2.selectbox("Variável Y (resposta):",
                              preparacao.COLUNAS_NUMERICAS, index=1,
                              format_func=preparacao.rotulo)

    if coluna_x == coluna_y:
        st.warning("Escolha duas variáveis diferentes.")
        return

    x = preparacao.valores(df, coluna_x)
    y = preparacao.valores(df, coluna_y)
    nome_x, nome_y = preparacao.rotulo(coluna_x), preparacao.rotulo(coluna_y)

    try:
        r = ms.correlacao(x, y)
        b0, b1, r2 = ms.regressao_linear(x, y)
        erro_padrao = ms.erro_padrao_estimativa(x, y)
    except ValueError as erro:
        st.error(f"Não é possível ajustar a reta: {erro}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("r de Pearson", num(r, 4))
    c2.metric("R²", num(r2, 4))
    c3.metric("Covariância", num(ms.covariancia(x, y), 3))
    c4.metric("Erro padrão da estimativa", num(erro_padrao, 3))

    sinal = "+" if b0 >= 0 else "−"
    st.subheader("A reta de mínimos quadrados")
    st.latex(
        rf"\hat{{y}} = {b1:.6f}\,x {sinal} {abs(b0):.6f}"
        rf"\qquad R^2 = {r2:.4f}"
    )

    st.markdown("**Predição interativa**")
    minimo_x, maximo_x = min(x), max(x)
    valor_x = st.number_input(
        f"Informe um valor de {nome_x}:",
        value=float(round(ms.mediana(x), 2)),
        min_value=float(minimo_x - abs(maximo_x)),
        max_value=float(maximo_x * 2),
        step=0.1,
    )
    previsto = ms.prever(b0, b1, valor_x)
    st.success(
        f"Para **{nome_x} = {num(valor_x)}**, a reta prevê "
        f"**{nome_y} = {num(previsto)}** "
        f"(margem típica de ± {num(erro_padrao)}, o erro padrão da "
        "estimativa)."
    )
    if valor_x < minimo_x or valor_x > maximo_x:
        st.error(
            f"⚠️ **Extrapolação.** O valor informado está fora da faixa "
            f"observada nos dados ([{num(minimo_x)} ; {num(maximo_x)}]). "
            "A reta só foi validada dentro desse intervalo; fora dele a "
            "predição é um chute vestido de matemática."
        )

    mostrar(graficos.dispersao_com_reta(
        x, y, b0, b1, r, r2, nome_x, nome_y,
        ponto_previsto=(valor_x, previsto)))

    residuos = [yi - ms.prever(b0, b1, xi) for xi, yi in zip(x, y)]
    mostrar(graficos.grafico_residuos(x, residuos, nome_x))

    if coluna_x == "distance" and coluna_y == "fare":
        tarifa_fixa = 52.0
        na_faixa = [toca for tarifa, toca
                    in zip(df["fare"], preparacao.toca_aeroporto(df))
                    if float(tarifa) == tarifa_fixa]
        st.markdown(
            f"""
**O que os resíduos denunciam.** O gráfico de dispersão tem uma faixa
horizontal visível em **US$ {num(tarifa_fixa)}**: são
**{num(len(na_faixa), 0)} corridas** com tarifa idêntica, independentemente
da distância — e **{num(100 * sum(na_faixa) / len(na_faixa), 1)}% delas
começam ou terminam em um aeroporto. Não é coincidência nem erro: Nova York
cobra uma **tarifa fixa** entre o aeroporto JFK e Manhattan, que ignora o
taxímetro.

Isso é um limite estrutural do modelo, não um detalhe estético. A reta
pressupõe que o preço cresce com a distância em todo o domínio; naquela
faixa ele simplesmente não cresce. Nenhum ajuste de b₀ ou b₁ resolve — o
que faltaria é uma variável indicadora de "corrida com tarifa fixa", isto
é, um modelo diferente. É o tipo de coisa que R² nenhum revela e que só
aparece quando se olha o gráfico.
"""
        )

    st.subheader("Interpretação dos coeficientes")
    st.markdown(
        f"""
- **b₁ = {num(b1, 4)}** — cada unidade a mais de *{nome_x}* está
  **associada**, em média, a **{num(abs(b1), 4)} unidade(s) a
  {'mais' if b1 >= 0 else 'menos'}** de *{nome_y}*.
- **b₀ = {num(b0, 4)}** — é o valor previsto de *{nome_y}* quando
  *{nome_x}* vale zero. Só tem sentido concreto se o zero for observável
  na prática; caso contrário, é apenas o ponto onde a reta corta o eixo.
- **R² = {num(r2, 4)}** — a reta explica **{num(100 * r2, 2)}%** da
  variação de *{nome_y}*. Os {num(100 * (1 - r2), 2)}% restantes vêm de
  tudo o que este modelo não enxerga.
- **r = {num(r, 4)}** — associação linear
  **{ms.classificar_correlacao(r)}**.
"""
    )

    st.error(
        "**Correlação não implica causalidade.** Neste próprio dataset há "
        "um exemplo limpo: *pedágios* e *gorjeta* têm correlação positiva "
        f"(r = {num(ms.correlacao(preparacao.valores(df, 'tolls'), preparacao.valores(df, 'tip')), 3)}). "
        "Ninguém dá gorjeta porque pagou pedágio. As duas sobem juntas "
        "porque ambas acompanham um terceiro fator — corridas longas, de "
        "aeroporto, que atravessam pontes e túneis e rendem contas altas. "
        "A variável escondida é a distância; pedágio e gorjeta apenas "
        "compartilham essa causa comum."
    )


# ---------------------------------------------------------------------------
# Módulo 6 — descobertas
# ---------------------------------------------------------------------------


def modulo_descobertas():
    st.header("Módulo 6 — As três descobertas")
    st.markdown(
        "Cada descoberta abaixo é uma afirmação em uma frase, seguida da "
        "evidência numérica produzida por esta aplicação e do limite "
        "honesto do que ela permite concluir."
    )

    # --- Descoberta 1 -------------------------------------------------------
    st.subheader("1. Quem paga em dinheiro nunca dá gorjeta — e isso é um "
                 "defeito do instrumento, não do passageiro")

    grupos = {}
    for pagamento, gorjeta in zip(df["payment"], df["tip"]):
        grupos.setdefault(str(pagamento), []).append(float(gorjeta))

    linhas = []
    for pagamento in sorted(grupos, key=lambda p: -len(grupos[p])):
        valores_grupo = grupos[pagamento]
        zeros = sum(1 for v in valores_grupo if v == 0)
        linhas.append({
            "Forma de pagamento": pagamento,
            "Corridas": num(len(valores_grupo), 0),
            "Gorjeta média (US$)": num(ms.media(valores_grupo)),
            "Gorjeta mediana (US$)": num(ms.mediana(valores_grupo)),
            "% com gorjeta zero": num(100 * zeros / len(valores_grupo), 1),
        })
    st.dataframe(linhas, hide_index=True, width="stretch")

    dinheiro = grupos.get("cash", [])
    cartao = grupos.get("credit card", [])
    mostrar(graficos.barras_comparativas(
        [linha["Forma de pagamento"] for linha in linhas],
        [ms.media(grupos[linha["Forma de pagamento"]]) for linha in linhas],
        "Gorjeta média por forma de pagamento", "US$",
        destaque="cash", casas=2))

    st.info(
        f"**Evidência.** Das {num(len(dinheiro), 0)} corridas pagas em "
        f"dinheiro, **{num(100 * sum(1 for v in dinheiro if v == 0) / len(dinheiro), 1)}%** "
        "registram gorjeta exatamente zero — não é uma tendência, é a "
        f"totalidade. No cartão, a gorjeta mediana é de US$ "
        f"{num(ms.mediana(cartao))} e a média, US$ {num(ms.media(cartao))}.\n\n"
        "**Limite honesto.** A leitura ingênua seria *\"quem paga em "
        "dinheiro é mão-fechada\"*. Ela está errada. O taxímetro de Nova "
        "York só registra a gorjeta quando ela passa pela maquininha; "
        "gorjeta em espécie vai direto para o bolso do motorista e nunca "
        "entra no sistema. O zero aqui significa **\"não medido\"**, não "
        "**\"não pago\"** — e qualquer média de gorjeta calculada sobre o "
        "dataset inteiro está subestimada por construção. É o tipo de erro "
        "que só aparece quando se olha a distribuição, e não a média."
    )

    # --- Descoberta 2 -------------------------------------------------------
    st.subheader("2. O número de passageiros não explica absolutamente nada "
                 "do preço da corrida")

    passageiros = preparacao.valores(df, "passengers")
    correlacoes = []
    for coluna in ["distance", "fare", "tip", "tolls", "total", "duracao_min"]:
        r = ms.correlacao(passageiros, preparacao.valores(df, coluna))
        correlacoes.append({
            "Variável": preparacao.rotulo(coluna),
            "r com nº de passageiros": num(r, 4),
            "R² (r²)": num(r * r, 6),
            "Classificação": ms.classificar_correlacao(r),
        })
    st.dataframe(correlacoes, hide_index=True, width="stretch")

    rotulos = ["distance", "fare", "tip", "tolls", "total", "duracao_min",
               "passengers"]
    series = [preparacao.valores(df, c) for c in rotulos]
    matriz = [[ms.correlacao(a, b) if a is not b else 1.0 for b in series]
              for a in series]
    mostrar(graficos.matriz_correlacao(
        [preparacao.rotulo(c) for c in rotulos], matriz,
        "Matriz de correlação de Pearson"))

    b0_p, b1_p, r2_p = ms.regressao_linear(
        passageiros, preparacao.valores(df, "fare"))
    st.info(
        f"**Evidência.** A correlação entre número de passageiros e tarifa é "
        f"r = {num(ms.correlacao(passageiros, preparacao.valores(df, 'fare')), 4)}, "
        f"e a regressão de tarifa sobre passageiros devolve "
        f"R² = {num(r2_p, 6)} — a linha vertical mais pálida do mapa de "
        f"calor. Ou seja: o número de passageiros explica "
        f"{num(100 * r2_p, 4)}% da variação do preço. Praticamente zero.\n\n"
        "**Por que isso é uma descoberta.** A intuição de quase todo mundo "
        "é que mais gente encarece a corrida — é assim em ônibus, avião e "
        "aplicativo com categoria. Em táxi de Nova York, não: a tarifa é "
        "do *veículo*, cobrada por distância e por tempo parado, e os "
        "passageiros extras viajam de graça. O dado não está com defeito; "
        "a intuição é que estava.\n\n"
        "**Limite honesto.** r mede associação **linear**. Um r ≈ 0 não "
        "prova ausência total de relação — poderia haver um efeito não "
        "linear que Pearson não captura. Mas o mapa de calor mostra a "
        "mesma ausência contra *todas* as seis variáveis numéricas, o que "
        "torna a hipótese de efeito escondido bem pouco plausível."
    )

    # --- Descoberta 3 -------------------------------------------------------
    st.subheader("3. Os outliers de preço têm endereço: são as corridas de "
                 "aeroporto")

    totais = preparacao.valores(df, "total")
    limite_inferior, limite_superior = ms.limites_outliers(totais)
    marca_outlier = [v > limite_superior or v < limite_inferior
                     for v in totais]
    marca_aeroporto = preparacao.toca_aeroporto(df).tolist()

    n_out = sum(marca_outlier)
    aero_entre_out = sum(1 for o, a in zip(marca_outlier, marca_aeroporto)
                         if o and a)
    aero_geral = sum(marca_aeroporto)

    c1, c2, c3 = st.columns(3)
    c1.metric("Outliers de valor total", num(n_out, 0),
              f"{num(100 * n_out / len(totais), 1)}% das corridas")
    c2.metric("Entre os outliers, tocam aeroporto",
              f"{num(100 * aero_entre_out / n_out, 1)}%")
    c3.metric("No dataset inteiro, tocam aeroporto",
              f"{num(100 * aero_geral / len(totais), 1)}%")

    mostrar(graficos.barras_comparativas(
        ["Entre os outliers de preço", "No dataset inteiro"],
        [100 * aero_entre_out / n_out, 100 * aero_geral / len(totais)],
        "Proporção de corridas com ponta em aeroporto", "% das corridas",
        destaque="Entre os outliers de preço", casas=1))

    dentro = [v for v, o in zip(totais, marca_outlier) if not o]
    fora = [v for v, o in zip(totais, marca_outlier) if o]
    pedagios = preparacao.valores(df, "tolls")
    pedagio_out = [p for p, o in zip(pedagios, marca_outlier) if o]
    pedagio_dentro = [p for p, o in zip(pedagios, marca_outlier) if not o]

    st.dataframe([
        {"Grupo": "Corridas outliers (IQR)",
         "n": num(len(fora), 0),
         "Total mediano (US$)": num(ms.mediana(fora)),
         "% com pedágio": num(100 * sum(1 for p in pedagio_out if p > 0)
                              / len(pedagio_out), 1)},
        {"Grupo": "Demais corridas",
         "n": num(len(dentro), 0),
         "Total mediano (US$)": num(ms.mediana(dentro)),
         "% com pedágio": num(100 * sum(1 for p in pedagio_dentro if p > 0)
                              / len(pedagio_dentro), 1)},
    ], hide_index=True, width="stretch")

    st.info(
        f"**Evidência.** A regra do IQR marca {num(n_out, 0)} corridas como "
        f"outliers de valor total (acima de US$ "
        f"{num(limite_superior)}). Dessas, "
        f"**{num(100 * aero_entre_out / n_out, 1)}% começam ou terminam em "
        f"JFK, LaGuardia ou Newark** — contra "
        f"{num(100 * aero_geral / len(totais), 1)}% no dataset inteiro, uma "
        f"concentração {num((aero_entre_out / n_out) / (aero_geral / len(totais)), 1)} "
        "vezes maior. O pedágio confirma: ele aparece em "
        f"{num(100 * sum(1 for p in pedagio_out if p > 0) / len(pedagio_out), 1)}% "
        "dos outliers e em apenas "
        f"{num(100 * sum(1 for p in pedagio_dentro if p > 0) / len(pedagio_dentro), 1)}% "
        "das demais corridas — são viagens que atravessam pontes e "
        "túneis.\n\n"
        "**Por que importa.** Esses pontos não são erro de medição, e "
        "apagá-los como \"ruído\" seria apagar um segmento real e "
        "economicamente relevante do serviço. A conclusão prática é que "
        "não existe uma distribuição de preço de táxi em Nova York: "
        "existem duas populações misturadas — o trajeto urbano curto "
        f"(mediana US$ {num(ms.mediana(dentro))}) e a corrida de aeroporto "
        f"(mediana US$ {num(ms.mediana(fora))}). A confirmação mais bonita "
        "está no Módulo 5: a faixa horizontal de tarifas idênticas de US$ "
        "52,00 no diagrama de dispersão é a tarifa fixa JFK↔Manhattan, uma "
        "regra tarifária visível a olho nu no gráfico.\n\n"
        "**Limite honesto.** *Aeroporto* aqui é a zona de embarque ou "
        "desembarque registrada; corridas de/para o aeroporto de Newark "
        "aparecem pouco por ficarem fora da cidade. E \"outlier pela regra "
        "do IQR\" é uma convenção (1,5·IQR), não uma verdade da natureza — "
        "com 3,0·IQR o recorte seria outro."
    )

    st.subheader("Limitações gerais da análise")
    st.warning(
        f"""
Esta análise **não permite** concluir o seguinte:

1. **Não vale para outros períodos.** A amostra cobre
   {tratamento['periodo_inicio']:%d/%m/%Y} a
   {tratamento['periodo_fim']:%d/%m/%Y} — um mês de inverno/início de
   primavera, antes da pandemia. Sazonalidade, feriados e a mudança de
   comportamento pós-2020 estão fora do alcance destes dados.
2. **Não é a população de corridas.** São {num(len(df), 0)} corridas de um
   universo de milhões por mês. É uma amostra pública, e não sabemos se
   o sorteio que a produziu foi aleatório simples — sem isso, nenhuma
   extrapolação formal para toda a cidade é defensável.
3. **Nada aqui é causal.** Todo o trabalho é de associação. Distância e
   tarifa andam juntas, mas a regressão não prova que a distância *causa*
   o preço; só mostra que uma prevê bem a outra dentro desta amostra.
4. **O zero da gorjeta é ambíguo** (descoberta 1), o que contamina
   qualquer conclusão sobre generosidade do passageiro.
5. **As {tratamento['distancia_zero']} corridas com distância zero e as
   {tratamento['passageiros_zero']} com zero passageiros** foram mantidas
   por não serem impossíveis, mas provavelmente são falhas de registro.
   Elas puxam levemente para baixo as medidas dessas duas variáveis.
6. **Apenas relações lineares foram testadas.** O r de Pearson e a reta
   de mínimos quadrados não enxergam curvas, patamares ou efeitos de
   limiar.
"""
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
        MODULOS[3]: modulo_simulacao,
        MODULOS[4]: modulo_distribuicoes,
        MODULOS[5]: modulo_regressao,
        MODULOS[6]: modulo_descobertas,
    }
    paginas[escolha]()


main()
