"""
gerar_figuras.py — produz as figuras e os números do relatório.

Roda fora do Streamlit, usando exatamente os mesmos módulos da aplicação
(`preparacao`, `minhastats`, `simulacao`, `graficos`). Isso garante que
cada número impresso no RELATORIO.md e no PDF é o mesmo que a aplicação
mostra na tela — não há uma segunda implementação em lugar nenhum.

Uso:  python gerar_figuras.py
Saída: arquivos PNG em assets/ e um resumo numérico no terminal.
"""

import json
import os

import graficos
import minhastats as ms
import preparacao
import simulacao

PASTA = "assets"
SEMENTE = 42


def salvar(fig, nome):
    caminho = os.path.join(PASTA, nome)
    fig.savefig(caminho, dpi=160, bbox_inches="tight",
                facecolor=graficos.SUPERFICIE)
    print(f"  figura: {caminho}")
    return caminho


def main():
    os.makedirs(PASTA, exist_ok=True)
    df, tratamento = preparacao.carregar()
    resultados = {"tratamento": {
        k: (str(v) if not isinstance(v, (int, float, dict)) else v)
        for k, v in tratamento.items()}}

    print(f"\n== Dataset ==")
    print(f"linhas originais : {tratamento['linhas_originais']}")
    print(f"linhas usadas    : {tratamento['linhas_finais']}")
    print(f"removidas        : {tratamento['removidas_duracao_invalida']}")
    print(f"periodo          : {tratamento['periodo_inicio']} a "
          f"{tratamento['periodo_fim']}")

    # ---------------------------------------------------------------- Módulo 2
    print("\n== Módulo 2 — descritiva ==")
    descritivas = {}
    for coluna in preparacao.COLUNAS_NUMERICAS:
        dados = preparacao.valores(df, coluna)
        resumo = ms.resumo_descritivo(dados)
        descritivas[coluna] = resumo
        print(f"\n{preparacao.rotulo(coluna)}")
        print(f"  n={resumo['n']}  media={resumo['media']:.4f}  "
              f"mediana={resumo['mediana']:.4f}  "
              f"moda={resumo['moda'][:3]}")
        print(f"  amplitude={resumo['amplitude']:.4f}  "
              f"var_amostral={resumo['variancia_amostral']:.4f}  "
              f"desvio={resumo['desvio_amostral']:.4f}  "
              f"cv={resumo['cv']:.2f}%")
        print(f"  Q1={resumo['q1']:.4f}  Q2={resumo['q2']:.4f}  "
              f"Q3={resumo['q3']:.4f}  IQR={resumo['iqr']:.4f}")
        print(f"  cercas=[{resumo['limite_inferior']:.2f} ; "
              f"{resumo['limite_superior']:.2f}]  "
              f"outliers={resumo['n_outliers']} "
              f"({resumo['pct_outliers']:.2f}%)")
        print(f"  assimetria g1={resumo['assimetria']:.4f}")
    resultados["descritivas"] = {
        c: {k: v for k, v in r.items() if k != "moda"}
        for c, r in descritivas.items()}

    total = preparacao.valores(df, "total")
    resumo_total = descritivas["total"]
    k_total = ms.numero_classes_sturges(len(total))
    classes_total = ms.tabela_frequencias_continua(total, k_total)
    salvar(graficos.histograma(
        total, classes_total, "Valor total pago por corrida (US$)",
        "total pago (US$)", media=resumo_total["media"],
        mediana=resumo_total["mediana"]), "fig02_histograma_total.png")
    salvar(graficos.boxplot(
        total, ms.outliers_iqr(total), resumo_total["limite_inferior"],
        resumo_total["limite_superior"],
        "Boxplot do valor total, com os outliers da regra do IQR",
        "total pago (US$)"), "fig02_boxplot_total.png")

    tabela_distrito = ms.tabela_frequencias_categorica(
        preparacao.categorias(df, "pickup_borough"))
    salvar(graficos.barras_categoricas(
        tabela_distrito, "Corridas por distrito de embarque"),
        "fig02_barras_distrito.png")
    print("\nDistrito de embarque:")
    for linha in tabela_distrito:
        print(f"  {linha['categoria']:<18} {linha['fi']:>5} "
              f"({100 * linha['fri']:.2f}%)")

    tabela_pagamento = ms.tabela_frequencias_categorica(
        preparacao.categorias(df, "payment"))
    print("Forma de pagamento:")
    for linha in tabela_pagamento:
        print(f"  {linha['categoria']:<18} {linha['fi']:>5} "
              f"({100 * linha['fri']:.2f}%)")

    # ---------------------------------------------------------------- Módulo 3
    print("\n== Módulo 3 — LGN e TCL ==")
    n_lgn = 5000
    lancamentos = simulacao.lancar_moeda(n_lgn, semente=SEMENTE)
    serie = simulacao.frequencia_relativa_acumulada(lancamentos, 1)
    salvar(graficos.grafico_lgn(
        serie, 0.5, f"Lei dos Grandes Números — {n_lgn} lançamentos de moeda",
        "cara"), "fig03_lgn.png")
    print(f"  freq. após 10={serie[9]:.4f}  100={serie[99]:.4f}  "
          f"1000={serie[999]:.4f}  {n_lgn}={serie[-1]:.4f}  "
          f"(teórica 0,5; erro final {abs(serie[-1] - 0.5):.4f})")
    resultados["lgn"] = {"n": n_lgn, "f10": serie[9], "f100": serie[99],
                         "f1000": serie[999], "final": serie[-1]}

    dados_tcl = preparacao.valores(df, "total")
    media_pop = ms.media(dados_tcl)
    desvio_pop = ms.desvio_padrao(dados_tcl, amostral=False)
    print(f"  população 'total': media={media_pop:.4f} "
          f"sigma={desvio_pop:.4f} g1={ms.assimetria(dados_tcl):.4f}")

    paineis, tabela_tcl = [], []
    for n in (2, 10, 30):
        medias = simulacao.medias_amostrais(dados_tcl, n, 2000,
                                            semente=SEMENTE)
        media_medias = ms.media(medias)
        desvio_medias = ms.desvio_padrao(medias, amostral=False)
        teorico = simulacao.erro_padrao_teorico(dados_tcl, n)
        # Sturges é a regra para uma TABELA legível; para julgar o FORMATO
        # de uma distribuição um grid mais fino comunica melhor.
        classes = ms.tabela_frequencias_continua(medias, 30)
        passo = (max(medias) - min(medias)) / 120
        curva_x = [min(medias) + i * passo for i in range(121)]
        curva_y = [ms.densidade_normal(v, media_medias, desvio_medias)
                   for v in curva_x]
        paineis.append({"n": n, "classes": classes, "curva_x": curva_x,
                        "curva_y": curva_y})
        tabela_tcl.append({"n": n, "media_das_medias": media_medias,
                           "desvio_observado": desvio_medias,
                           "sigma_sobre_raiz_n": teorico,
                           "assimetria": ms.assimetria(medias)})
        print(f"  n={n:<3} média das médias={media_medias:.4f}  "
              f"desvio obs={desvio_medias:.4f}  σ/√n={teorico:.4f}  "
              f"g1={ms.assimetria(medias):.4f}")
    salvar(graficos.grafico_tcl(
        paineis, "Teorema Central do Limite sobre o valor total das corridas "
                 "(2.000 amostras por painel)"), "fig03_tcl.png")
    resultados["tcl"] = tabela_tcl

    # ---------------------------------------------------------------- Módulo 4
    print("\n== Módulo 4 — distribuições teóricas ==")
    distancia = preparacao.valores(df, "distance")
    media_d = ms.media(distancia)
    desvio_d = ms.desvio_padrao(distancia)
    lam_exp = 1 / media_d
    # 40 classes (e não Sturges): aqui o histograma não é uma tabela de
    # leitura, é o objeto que vai ser comparado com uma curva contínua —
    # precisa de resolução para que o ajuste possa ser julgado.
    classes_d = ms.tabela_frequencias_continua(distancia, 40)
    passo = (max(distancia) - min(distancia)) / 300
    xs = [min(distancia) + i * passo for i in range(301)]
    curvas = [
        (f"Normal(μ = {media_d:.2f}; σ = {desvio_d:.2f})".replace(".", ","),
         xs, [ms.densidade_normal(v, media_d, desvio_d) for v in xs]),
        (f"Exponencial(λ = {lam_exp:.4f})".replace(".", ","),
         xs, [ms.densidade_exponencial(v, lam_exp) for v in xs]),
    ]
    salvar(graficos.histograma_com_curvas(
        classes_d, curvas,
        "Distância das corridas: Normal e Exponencial estimadas dos dados",
        "distância (milhas)"), "fig04_distancia_ajuste.png")

    erros = {}
    for rotulo_curva, _, _ in curvas:
        if rotulo_curva.startswith("Normal"):
            f = lambda v: ms.densidade_normal(v, media_d, desvio_d)
        else:
            f = lambda v: ms.densidade_exponencial(v, lam_exp)
        lista = [abs(c["densidade"] - f(c["ponto_medio"])) for c in classes_d]
        erros[rotulo_curva] = {"erro_medio": ms.media(lista),
                               "erro_max": max(lista)}
        print(f"  {rotulo_curva}: EMA={ms.media(lista):.6f} "
              f"máx={max(lista):.6f}")
    print(f"  distância: média={media_d:.4f} desvio={desvio_d:.4f} "
          f"g1={ms.assimetria(distancia):.4f} λ̂={lam_exp:.4f}")
    resultados["ajuste_distancia"] = erros

    passageiros = [int(v) for v in preparacao.valores(df, "passengers")]
    lam_poisson = ms.media(passageiros)
    tabela_p = sorted(ms.tabela_frequencias_categorica(passageiros),
                      key=lambda linha: linha["categoria"])
    categorias_p = [linha["categoria"] for linha in tabela_p]
    observado = [linha["fri"] for linha in tabela_p]
    teorico = [ms.probabilidade_poisson(k, lam_poisson) for k in categorias_p]
    salvar(graficos.barras_observado_teorico(
        categorias_p, observado, teorico,
        f"Poisson(λ = {lam_poisson:.3f})".replace(".", ","),
        "Número de passageiros: observado × Poisson", "passageiros"),
        "fig04_poisson_passageiros.png")
    print(f"  passageiros: λ̂={lam_poisson:.4f}")
    for k, o, t in zip(categorias_p, observado, teorico):
        print(f"    k={k}: observado={100 * o:6.2f}%  Poisson={100 * t:6.2f}%")
    resultados["poisson"] = {"lambda": lam_poisson,
                             "observado": dict(zip(map(str, categorias_p),
                                                   observado)),
                             "teorico": dict(zip(map(str, categorias_p),
                                                 teorico))}

    # ---------------------------------------------------------------- Módulo 5
    print("\n== Módulo 5 — correlação e regressão ==")
    x = preparacao.valores(df, "distance")
    y = preparacao.valores(df, "fare")
    r = ms.correlacao(x, y)
    b0, b1, r2 = ms.regressao_linear(x, y)
    erro_padrao = ms.erro_padrao_estimativa(x, y)
    previsao_x = 5.0
    previsao_y = ms.prever(b0, b1, previsao_x)
    salvar(graficos.dispersao_com_reta(
        x, y, b0, b1, r, r2, "distância (milhas)", "tarifa (US$)",
        ponto_previsto=(previsao_x, previsao_y)), "fig05_regressao.png")
    residuos = [yi - ms.prever(b0, b1, xi) for xi, yi in zip(x, y)]
    salvar(graficos.grafico_residuos(x, residuos, "distância (milhas)"),
           "fig05_residuos.png")
    print(f"  distância × tarifa: r={r:.6f}  b1={b1:.6f}  b0={b0:.6f}  "
          f"R²={r2:.6f}  cov={ms.covariancia(x, y):.4f}  "
          f"se={erro_padrao:.4f}")
    print(f"  predição x=5 milhas -> ŷ={previsao_y:.4f}")
    print(f"  faixa observada de x: [{min(x):.2f} ; {max(x):.2f}]")
    r_tolls_tip = ms.correlacao(preparacao.valores(df, "tolls"),
                                preparacao.valores(df, "tip"))
    print(f"  pedágio × gorjeta (correlação espúria): r={r_tolls_tip:.4f}")

    # A faixa horizontal visível no gráfico: a tarifa fixa JFK <-> Manhattan.
    toca_aero = preparacao.toca_aeroporto(df).tolist()
    faixa = [a for tarifa, a in zip(df["fare"], toca_aero)
             if float(tarifa) == 52.0]
    pct_faixa = 100 * sum(faixa) / len(faixa)
    print(f"  tarifa fixa US$ 52,00: {len(faixa)} corridas, "
          f"{pct_faixa:.1f}% tocam aeroporto")
    resultados["regressao"] = {"r": r, "b0": b0, "b1": b1, "r2": r2,
                               "se": erro_padrao,
                               "cov": ms.covariancia(x, y),
                               "r_tolls_tip": r_tolls_tip,
                               "tarifa_fixa_n": len(faixa),
                               "tarifa_fixa_pct_aero": pct_faixa,
                               "previsao_5_milhas": previsao_y}

    # ---------------------------------------------------------------- Módulo 6
    print("\n== Módulo 6 — descobertas ==")

    grupos = {}
    for pagamento, gorjeta in zip(df["payment"], df["tip"]):
        grupos.setdefault(str(pagamento), []).append(float(gorjeta))
    nomes = sorted(grupos, key=lambda p: -len(grupos[p]))
    salvar(graficos.barras_comparativas(
        nomes, [ms.media(grupos[nome]) for nome in nomes],
        "Gorjeta média por forma de pagamento", "US$", destaque="cash"),
        "fig06_gorjeta_pagamento.png")
    descoberta1 = {}
    for nome in nomes:
        valores_grupo = grupos[nome]
        zeros = sum(1 for v in valores_grupo if v == 0)
        descoberta1[nome] = {
            "n": len(valores_grupo), "media": ms.media(valores_grupo),
            "mediana": ms.mediana(valores_grupo),
            "pct_zero": 100 * zeros / len(valores_grupo)}
        print(f"  {nome:<16} n={len(valores_grupo):>5} "
              f"média={ms.media(valores_grupo):.4f} "
              f"mediana={ms.mediana(valores_grupo):.2f} "
              f"zeros={100 * zeros / len(valores_grupo):.2f}%")
    resultados["descoberta1"] = descoberta1

    colunas_matriz = ["distance", "fare", "tip", "tolls", "total",
                      "duracao_min", "passengers"]
    series = [preparacao.valores(df, c) for c in colunas_matriz]
    matriz = [[1.0 if i == j else ms.correlacao(a, b)
               for j, b in enumerate(series)]
              for i, a in enumerate(series)]
    salvar(graficos.matriz_correlacao(
        [preparacao.rotulo(c) for c in colunas_matriz], matriz,
        "Matriz de correlação de Pearson"), "fig06_matriz_correlacao.png")
    passageiros_f = preparacao.valores(df, "passengers")
    print("  correlações com nº de passageiros:")
    descoberta2 = {}
    for coluna in ["distance", "fare", "tip", "tolls", "total", "duracao_min"]:
        rr = ms.correlacao(passageiros_f, preparacao.valores(df, coluna))
        descoberta2[coluna] = rr
        print(f"    {coluna:<12} r={rr:+.4f}  r²={rr * rr:.6f}")
    _, _, r2_pass = ms.regressao_linear(passageiros_f,
                                        preparacao.valores(df, "fare"))
    print(f"    R² da regressão passageiros -> tarifa: {r2_pass:.8f}")
    resultados["descoberta2"] = {"correlacoes": descoberta2,
                                 "r2_passageiros_tarifa": r2_pass}

    limite_inf, limite_sup = ms.limites_outliers(total)
    marca_outlier = [v > limite_sup or v < limite_inf for v in total]
    marca_aero = preparacao.toca_aeroporto(df).tolist()
    n_out = sum(marca_outlier)
    aero_out = sum(1 for o, a in zip(marca_outlier, marca_aero) if o and a)
    aero_total = sum(marca_aero)
    pct_out = 100 * aero_out / n_out
    pct_geral = 100 * aero_total / len(total)
    salvar(graficos.barras_comparativas(
        ["Entre os outliers de preço", "No dataset inteiro"],
        [pct_out, pct_geral],
        "Proporção de corridas com ponta em aeroporto", "% das corridas",
        destaque="Entre os outliers de preço", casas=1),
        "fig06_outliers_aeroporto.png")
    dentro = [v for v, o in zip(total, marca_outlier) if not o]
    fora = [v for v, o in zip(total, marca_outlier) if o]
    pedagios = preparacao.valores(df, "tolls")
    ped_out = [p for p, o in zip(pedagios, marca_outlier) if o]
    ped_dentro = [p for p, o in zip(pedagios, marca_outlier) if not o]
    print(f"  outliers de total: {n_out} ({100 * n_out / len(total):.2f}%)  "
          f"limite superior US$ {limite_sup:.2f}")
    print(f"  aeroporto entre outliers: {pct_out:.2f}%  "
          f"no dataset: {pct_geral:.2f}%  razão {pct_out / pct_geral:.2f}x")
    print(f"  mediana total: outliers US$ {ms.mediana(fora):.2f}  "
          f"demais US$ {ms.mediana(dentro):.2f}")
    print(f"  com pedágio: outliers "
          f"{100 * sum(1 for p in ped_out if p > 0) / len(ped_out):.2f}%  "
          f"demais "
          f"{100 * sum(1 for p in ped_dentro if p > 0) / len(ped_dentro):.2f}%")
    resultados["descoberta3"] = {
        "n_outliers": n_out, "pct_outliers": 100 * n_out / len(total),
        "limite_superior": limite_sup, "pct_aero_outliers": pct_out,
        "pct_aero_geral": pct_geral, "razao": pct_out / pct_geral,
        "mediana_outliers": ms.mediana(fora),
        "mediana_demais": ms.mediana(dentro),
        "pct_pedagio_outliers":
            100 * sum(1 for p in ped_out if p > 0) / len(ped_out),
        "pct_pedagio_demais":
            100 * sum(1 for p in ped_dentro if p > 0) / len(ped_dentro)}

    caminho = os.path.join(PASTA, "resultados.json")
    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(resultados, arquivo, ensure_ascii=False, indent=2,
                  default=str)
    print(f"\nResumo numérico salvo em {caminho}")


if __name__ == "__main__":
    main()
