# deep-learning-pa1

Programming Assignment 1 de Deep Learning, Ciencia de Dados FGV. A ideia do PA e
fazer as arquiteturas encoder-decoder que a gente viu na aula de segmentacao
semantica (FCN, SegNet, U-Net, ResUNet, DeepLab, PSPNet) produzirem rotulo
instance-aware, sem usar detector com proposta de regiao. Nada de Mask R-CNN,
detectron2, SAM, Cellpose ou StarDist.

Dataset e o DSB2018 / BBBC038v1 (opcao A do enunciado), microscopia de nucleo com
uma mascara PNG por nucleo. Escolhemos ele porque baixa direto sem conta, nao tem
anotacao COCO pra parsear e nucleo encostado e a regra e nao a excecao, que e
exatamente o problema que o PA quer que a gente ataque.

Da Parte 2 a gente escolheu a trilha A, fronteira e watershed. A rede continua sendo
a mesma U-Net com encoder ResNet, muda so o que ela preve: em vez de 1 canal de
foreground, sao 3 classes (fundo, interior, fronteira entre instancias) mais um mapa
continuo de distancia ao fundo. A decodificacao vira watershed com os interiores como
marcador. Na Parte 3 rodamos os eixos 2 (funcao de perda) e 3 (contexto global).

## Ambiente

Usa uv. Depois de clonar:

```
uv sync
```

Isso ja instala torch, torchvision, albumentations, scikit-image, scikit-learn e o
resto. O codigo escolhe o device sozinho: cuda se tiver, senao mps (mac), senao cpu.

## Dados

```
uv run python scripts/download_dsb2018.py
```

Baixa o stage1_train do site do Broad (83 MB) e extrai em `data/dsb2018/`. Sao 670
imagens.

O split treino/val/teste e estratificado por modalidade, como o enunciado exige. O
DSB2018 nao tem essa coluna, entao a gente descreve cada imagem com estatistica de cor
(media e desvio do cinza, saturacao, fracao de pixel muito escuro e muito claro,
diferenca entre canais) e roda um k-means com 4 clusters em cima disso. Na pratica os
clusters separam fluorescencia de fundo preto, brightfield e histologia. O split e
feito dentro de cada cluster, entao as modalidades aparecem nas mesmas proporcoes nos
tres conjuntos. Ta em `src/pa1/data/splits.py` e o resultado fica cacheado em
`data/dsb2018/modality.json`.

O dataset sintetico da Parte 0 nao precisa baixar nada, ele gera na hora a partir de
uma seed por indice.

## Treinar

Um comando treina:

```
uv run python scripts/train.py --config configs/dsb2018_boundary.yaml
```

Esse e o modelo final (Parte 2). O baseline da Parte 1 e
`configs/dsb2018_baseline.yaml`. Pro teste unitario sintetico da Parte 0, que roda em
menos de 5 minutos, e `configs/synthetic_baseline.yaml` (versao binaria) ou
`configs/synthetic_boundary.yaml` (mesma coisa ja com a cabeca de instancias).

O checkpoint salvo e o de melhor mAP de validacao, nao o de menor loss. Isso importa
porque da pra ter Dice quase 1 e mAP de instancia pessimo, que e justamente o que a
Parte 1 quer mostrar.

## Avaliar

Um comando avalia:

```
uv run python scripts/evaluate.py --config configs/dsb2018_boundary.yaml --checkpoint runs/dsb2018_boundary/best.pt
```

Imprime IoU e Dice semanticos, mAP@[.50:.95], AP@.50 e erro absoluto medio de
contagem, e salva em `runs/<nome>/eval_test/` um metrics.json com a tabela por imagem,
o grafico de mAP e erro de contagem contra densidade de objetos (item 5 da Parte 1) e
alguns paineis qualitativos.

O matching e implementado por nos em `src/pa1/metrics/instance.py`, guloso por IoU
decrescente por padrao. Da pra trocar por Hungarian com `--matching hungarian`.
Precisao num limiar e TP / (TP + FP + FN), que e a convencao do DSB2018, e o AP da
imagem e a media sobre os limiares de 0.50 a 0.95 de 0.05 em 0.05.

## Reproduzir cada parte

```
# Parte 0, teste unitario sintetico
uv run python scripts/train.py --config configs/synthetic_boundary.yaml

# Parte 1, baseline binario + instancias por limiar e componentes conexos
uv run python scripts/train.py    --config configs/dsb2018_baseline.yaml
uv run python scripts/evaluate.py --config configs/dsb2018_baseline.yaml --checkpoint runs/dsb2018_baseline/best.pt

# Parte 2, trilha A, e a comparacao lado a lado com a Parte 1
uv run python scripts/class_stats.py --config configs/dsb2018_boundary.yaml
uv run python scripts/train.py       --config configs/dsb2018_boundary.yaml
uv run python scripts/compare.py --baseline configs/dsb2018_baseline.yaml runs/dsb2018_baseline/best.pt \
                                 --instance configs/dsb2018_boundary.yaml runs/dsb2018_boundary/best.pt

# Parte 3, eixos 2 e 3, 2 seeds cada configuracao
uv run python scripts/ablations.py --config configs/dsb2018_boundary.yaml --axis 2 --seeds 0 1
uv run python scripts/ablations.py --config configs/dsb2018_boundary.yaml --axis 3 --seeds 0 1

# Partes 4, 5 e 6
uv run python scripts/part4_mosaic.py   --config configs/dsb2018_boundary.yaml --checkpoint runs/dsb2018_boundary/best.pt
uv run python scripts/part5_failures.py --config configs/dsb2018_boundary.yaml --checkpoint runs/dsb2018_boundary/best.pt
uv run python scripts/part6_stress.py   --config configs/dsb2018_boundary.yaml --checkpoint runs/dsb2018_boundary/best.pt
```

Os numeros da apresentacao estao todos em APRESENTACAO.md, cada um apontando pro
arquivo em `runs/` de onde saiu.

## Inferencia

`inferencia.ipynb` recebe o caminho de uma imagem qualquer e devolve a mascara de
instancias colorida e a contagem, sem retreinar. Ele so carrega
`runs/dsb2018_boundary/best.pt`. A logica ta em `src/pa1/inference.py`, e ela decide
sozinha se roda a imagem inteira ou em tiles (acima de 768 px de lado passa pro modo
tiled com media de logits, que foi a conclusao da Parte 4).

## Testes

```
uv run pytest
```

Cobre o gerador sintetico, as metricas de instancia com caso montado na mao (match
perfeito da mAP 1, um FN da 0.5, um FP da 2/3, dois objetos fundidos num blob so da
menos de 0.5), os alvos da trilha A e a fusao de instancias entre tiles da Parte 4.

O teste que mais vale a pena olhar e o do teto de oraculo em
`tests/test_targets_and_postprocess.py`: com a fronteira e a distancia perfeitas o
watershed da mAP acima de 0.70 no sintetico, enquanto componentes conexos com mascara
semantica perfeita nao passa de 0.11. Esse numero e o argumento inteiro da Parte 2, e
da pra checar sem treinar nada.

## Onde esta cada parte

```
src/pa1/data/synthetic.py     Parte 0, gerador de elipses
src/pa1/data/splits.py        split estratificado por modalidade
src/pa1/data/targets.py       Parte 2, rotulo de fronteira, mapa de distancia, pesos
src/pa1/models/unet.py        encoder-decoder, o mesmo nas duas partes
src/pa1/models/context.py     Parte 3 eixo 3, ParseNet e PSPNet
src/pa1/losses.py             CE, CE balanceada, focal, L1/L2
src/pa1/postprocess/naive.py  Parte 1, limiar + componentes conexos
src/pa1/postprocess/watershed.py  Parte 2, watershed com marcadores
src/pa1/metrics/instance.py   matching e mAP, escritos por nos
src/pa1/tiling.py             Parte 4, tiles e fusao de instancias
src/pa1/receptive_field.py    Parte 5, campo receptivo teorico
src/pa1/corruptions.py        Parte 6, blur, ruido, brilho/contraste
```

## Onde os treinos rodaram

Nao tem GPU nas nossas maquinas, entao os treinos sairam em kernel do Kaggle com T4.
`notebooks/colab_setup.ipynb` faz a mesma coisa no Colab. Tudo roda em CPU tambem, so
demora: o baseline do DSB2018 leva algumas horas em vez de alguns minutos.
