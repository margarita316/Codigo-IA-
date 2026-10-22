# Comparação padronizada de SSM-DTA, SubMDTA e ensemble na predição de afinidade fármaco-alvo (DAVIS)

Documentos na raiz do repositório: Relatorio_IC_DAVIS_SSMDTA_SubMDTA_ABNT.pdf (relatório técnico, ABNT) e Material_de_Apoio_EICTI2026.pdf (material de apoio da apresentação).

Material complementar do trabalho **"Predição de propriedades de interações entre fármacos e receptores por meio de algoritmos de inteligência artificial"**, apresentado no XV Encontro Anual de Iniciação Científica e XI Encontro de Iniciação em Desenvolvimento Tecnológico e Inovação (EICTI), 8ª SIEPE, UNILA.

Este diretório reúne, em um único lugar, os documentos, os códigos, os dados de partição, as predições e as métricas usadas no estudo, para que a comparação possa ser verificada.

> Esta pasta contém apenas o estudo de 2026. Os arquivos publicados anteriormente neste repositório estão separados em [`../material_anterior/`](../material_anterior/) e não fazem parte dos resultados abaixo.

## Resumo do estudo

Dois métodos da literatura de predição de afinidade fármaco-alvo (DTI) foram avaliados **sob condições idênticas**, mantendo suas arquiteturas originais inalteradas:

- **SSM-DTA** (variante *baseline*, sem pré-treinamento MLM);
- **SubMDTA** (GINConvNet + GraphEnhance, com o *checkpoint* `sub_50000.pth` do repositório original).

Também foi construído um **ensemble simples**, a média das predições par a par dos dois métodos.

| Item | Configuração |
|---|---|
| Banco de dados | DAVIS (versão oficial, afinidade expressa como pKd = 9 − log10(Kd em nM)) |
| Amostra | 150 interações fármaco-alvo extraídas aleatoriamente |
| Validação | validação cruzada com 5 folds (mesmas partições para os dois métodos) |
| Partição por fold | 30 pares de teste, 108 de treino e 12 de validação |
| Treinamento | 25 épocas por fold |
| Ambiente | Python 3.7, execução local |
| Métrica principal | RMSE no conjunto de teste de cada fold |

A partição é feita **por pares** fármaco-alvo (não é uma partição por fármaco ou por proteína novos).

## Resultados

RMSE por fold (menor é melhor):

| Fold | SSM-DTA | SubMDTA | Ensemble |
|---|---|---|---|
| 0 | 0,636 | 0,943 | 0,739 |
| 1 | 0,731 | 1,307 | 0,912 |
| 2 | 0,577 | 0,473 | **0,415** |
| 3 | 1,251 | 1,420 | 1,273 |
| 4 | 1,186 | 1,488 | 1,293 |
| **Média ± desvio padrão** | **0,876 ± 0,284** | 1,126 ± 0,377 | 0,926 ± 0,332 |

O desvio padrão é populacional (ddof = 0), calculado sobre os 5 folds.

Comparações pareadas do RMSE (n = 5 folds):

| Comparação | Dif. média | p (teste t) | p (Wilcoxon) | d de Cohen |
|---|---|---|---|---|
| SSM-DTA vs. SubMDTA | −0,250 | 0,086 | 0,125 | −1,01 |
| SSM-DTA vs. Ensemble | −0,050 | 0,444 | 0,438 | −0,38 |
| SubMDTA vs. Ensemble | 0,200 | 0,022 | 0,063 | 1,62 |

Com apenas 5 folds, o menor valor possível do p do teste de Wilcoxon é 0,0625; por isso esse teste não pode indicar significância neste desenho. O ensemble superou significativamente o SubMDTA (teste t pareado), mas não o SSM-DTA, que teve o melhor desempenho médio. O ensemble foi o melhor dos três apenas no fold 2.

> **Limitações.** A amostra é pequena (150 pares), o número de folds é reduzido (5) e a partição é aleatória por pares. Os resultados descrevem o comportamento sob esse cenário específico e não devem ser tomados como ranking geral dos métodos.

## Estrutura do diretório

```
estudo_davis150_5folds/
├── README.md
├── documentos/
│   ├── resumo_submetido_EICTI2026.pdf     resumo submetido ao evento
│   └── apresentacao_EICTI2026.pdf         apresentação (PDF)
├── codigo/
│   ├── 01_preparacao_dados/prepare_davis_cv.py
│   ├── 02_ssm_dta/                        run_cv_training.py, extract_predictions.py
│   ├── 03_submdta/run_submdta_cv.py
│   ├── 04_ensemble_e_estatistica/         ensemble_results.py, stat_analysis.py,
│   │                                      per_fold_metrics.py, full_metrics.py
│   └── 05_figuras/                        plot_results2.py, plot_model_comparison.py
├── dados/
│   └── davis150_folds/fold0 ... fold4/    partições (treino, validação e teste)
└── resultados/
    ├── predicoes/ssm_dta/                 predições de teste por fold
    ├── predicoes/submdta/                 predições e rótulos de teste por fold
    ├── metricas/                          resumos em JSON
    └── figuras/                           figuras dos resultados
```

## Como reproduzir

**Verificação rápida (sem treinar).** Com Python 3 e `numpy` e `scipy` instalados, as métricas, o ensemble e a análise estatística são recalculados a partir das predições incluídas:

```bash
cd codigo/04_ensemble_e_estatistica
python per_fold_metrics.py      # RMSE, Pearson e C-index por fold
python ensemble_results.py      # ensemble e resumo (resultados/metricas/ensemble_summary.json)
python stat_analysis.py         # testes pareados (resultados/metricas/stat_analysis_summary.json)
python full_metrics.py          # tabela de métricas com média ± desvio padrão
```

Os scripts usam caminhos relativos à raiz deste diretório.

**Reprodução completa (treino).**

1. Obter os códigos originais dos métodos (veja a seção seguinte) e instalar o ambiente de cada um.
2. Obter o DAVIS e gerar as partições: `codigo/01_preparacao_dados/prepare_davis_cv.py`. As partições usadas já estão em `dados/davis150_folds/`.
3. SSM-DTA: `codigo/02_ssm_dta/run_cv_training.py` (treino e teste por fold) e `extract_predictions.py` (predições por par).
4. SubMDTA: `codigo/03_submdta/run_submdta_cv.py`, que reutiliza as mesmas partições (arquitetura inalterada; número de épocas reduzido para 25).
5. Ensemble e estatística: scripts de `codigo/04_ensemble_e_estatistica/`.

Os scripts de treino foram escritos para serem executados dentro da pasta do repositório original de cada método (`SSM-DTA` e `SubMDTA`). Os caminhos padrão são relativos e podem ser alterados por argumentos de linha de comando.

## Códigos e dados de terceiros

Este diretório **não redistribui** os repositórios originais, apenas os scripts escritos para este estudo.

- **SSM-DTA**: Pei, Q. et al. Repositório original: <https://github.com/QizhiPei/SSM-DTA> (licença MIT). Foram usados com ele o `fairseq` 0.10.2 e a variante *baseline*.
- **SubMDTA**: SubMDTA, "Drug target affinity prediction based on substructure extraction and multi-scale features". O repositório original não declara licença; por isso seu código e seu *checkpoint* não foram incluídos aqui.
- **DAVIS**: Davis, M. I. et al. Comprehensive analysis of kinase inhibitor selectivity. *Nature Biotechnology*, v. 29, p. 1046-1051, 2011. Os 150 pares usados estão em `dados/davis150_folds/`, como derivados do conjunto público.

Pequenos ajustes de compatibilidade feitos para executar o código original no ambiente local (Windows, Python 3.7): remoção de um bloco de *symlink* no `setup.py` do SSM-DTA e ajustes de versão das dependências do SubMDTA (`torch-geometric` 2.0.4 e `torch-sparse` 0.6.12 com `torch` 1.10.2). As arquiteturas dos modelos não foram modificadas.

## Referências

1. CHACON ROMERO, Margarita; MACIEL, Joylan Nunes; ZALEWSKI, Willian. Inteligência artificial para a predição de interações entre fármacos e receptores. In: SEMANA INTEGRADA DE ENSINO, PESQUISA E EXTENSÃO, 7., 2025, Foz do Iguaçu. *Anais* [...]. Foz do Iguaçu: UNILA, 2025. p. 132-133.
2. CHACON ROMERO, Margarita. *Código IA*: códigos para predição de interações fármaco-receptor. [S. l.]: GitHub, 2026. Disponível em: https://github.com/margarita316/Codigo-IA-.

## Autoria e agradecimentos

Margarita Chacon Romero (bolsista IC, PRPPG/UNILA, Engenharia Química), com coorientação de Joylan Nunes Maciel e orientação de Willian Zalewski (UNILA). Os métodos foram implementados e executados localmente, com auxílio do Claude (Anthropic). Agradecimento à PRPPG/UNILA pelo apoio financeiro por meio da bolsa de pesquisa.
