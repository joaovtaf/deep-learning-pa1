# deep-learning-pa1

Programming Assignment 1 de Deep Learning, Ciência de Dados FGV. A ideia do PA é
fazer as arquiteturas encoder-decoder que a gente viu na aula de segmentação
semântica (FCN, SegNet, U-Net, ResUNet, DeepLab, PSPNet) produzirem rótulo
instance-aware, sem usar detector com proposta de região. Nada de Mask R-CNN,
detectron2, SAM, Cellpose ou StarDist.

O dataset é o DSB2018 / BBBC038v1 (opção A do enunciado), microscopia de núcleo com
uma máscara PNG por núcleo. Escolhemos ele porque baixa direto sem conta, não tem
anotação COCO pra parsear e núcleo encostado é a regra e não a exceção, que é
exatamente o problema que o PA quer que a gente ataque.

Da Parte 2 a gente escolheu a trilha A, fronteira e watershed. A rede continua sendo
a mesma U-Net com encoder ResNet, muda só o que ela prevê: em vez de 1 canal de
foreground, são 3 classes (fundo, interior, fronteira entre instâncias) mais um mapa
contínuo de distância ao fundo. A decodificação vira watershed com os interiores como
marcador. Na Parte 3 rodamos os eixos 2 (função de perda) e 3 (contexto global).

## Ambiente

Usa uv. Depois de clonar:

```
uv sync
```

Isso já instala torch, torchvision, albumentations, scikit-image, scikit-learn e o
resto. O código escolhe o device sozinho: cuda se tiver, senão mps (mac), senão cpu.

## Dados

```
uv run python scripts/download_dsb2018.py
```

Baixa o stage1_train do site do Broad (83 MB) e extrai em `data/dsb2018/`. São 670
imagens.

O split treino/val/teste é estratificado por modalidade, como o enunciado exige. O
DSB2018 não tem essa coluna, então a gente descreve cada imagem com estatística de cor
(média e desvio do cinza, saturação, fração de pixel muito escuro e muito claro,
diferença entre canais) e roda um k-means com 4 clusters em cima disso. Na prática os
clusters separam fluorescência de fundo preto, brightfield e histologia. O split é
feito dentro de cada cluster, então as modalidades aparecem nas mesmas proporções nos
três conjuntos. Está em `src/pa1/data/splits.py` e o resultado fica cacheado em
`data/dsb2018/modality.json`.

O dataset sintético da Parte 0 não precisa baixar nada, ele gera na hora a partir de
uma seed por índice.

## Treinar

Um comando treina:

```
uv run python scripts/train.py --config configs/dsb2018_boundary.yaml
```

Esse é o modelo final (Parte 2). O baseline da Parte 1 é
`configs/dsb2018_baseline.yaml`. Pro teste unitário sintético da Parte 0, que roda em
menos de 5 minutos, é `configs/synthetic_baseline.yaml` (versão binária) ou
`configs/synthetic_boundary.yaml` (mesma coisa já com a cabeça de instâncias).

O checkpoint salvo é o de melhor mAP de validação, não o de menor loss. Isso importa
porque dá pra ter Dice quase 1 e mAP de instância péssimo, que é justamente o que a
Parte 1 quer mostrar.

Depois de treinar, vale calibrar a decodificação na validação antes de medir no teste:

```
uv run python scripts/tune_watershed.py --config configs/dsb2018_boundary.yaml --checkpoint runs/dsb2018_boundary/best.pt
```

Só isso vale uns 0.08 de mAP nos dois modelos, sem mexer em peso nenhum. Os limiares que
saíram dessa varredura já estão nos configs `dsb2018_final*.yaml`.

## Avaliar

Um comando avalia:

```
uv run python scripts/evaluate.py --config configs/dsb2018_boundary.yaml --checkpoint runs/dsb2018_boundary/best.pt
```

Imprime IoU e Dice semânticos, mAP@[.50:.95], AP@.50 e erro absoluto médio de
contagem, e salva em `runs/<nome>/eval_test/` um metrics.json com a tabela por imagem,
o gráfico de mAP e erro de contagem contra densidade de objetos (item 5 da Parte 1) e
alguns painéis qualitativos.

O matching é implementado por nós em `src/pa1/metrics/instance.py`, guloso por IoU
decrescente por padrão. Dá pra trocar por Hungarian com `--matching hungarian`.
Precisão num limiar é TP / (TP + FP + FN), que é a convenção do DSB2018, e o AP da
imagem é a média sobre os limiares de 0.50 a 0.95 de 0.05 em 0.05.

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

Os números da apresentação estão todos em APRESENTACAO.md, cada um apontando pro
arquivo em `results/` de onde saiu.

## Checkpoint

O peso do modelo final tem 93 MB, que não cabe confortavelmente num commit, então ele está
como dataset privado do Kaggle:

    https://www.kaggle.com/datasets/joaovtaf/pa1-checkpoint

Baixa e põe em `runs/dsb2018_boundary/best.pt` (o `dsb2018_baseline.pt` do mesmo dataset é
o baseline da Parte 1, se quiser reproduzir a comparação). Ou treina do zero, que leva 6
minutos numa GPU.

    kaggle datasets download -d joaovtaf/pa1-checkpoint -p runs --unzip

## Inferência

`inferencia.ipynb` recebe o caminho de uma imagem qualquer e devolve a máscara de
instâncias colorida e a contagem, sem retreinar. Ele só carrega
`runs/dsb2018_boundary/best.pt`. A lógica está em `src/pa1/inference.py`, e ela decide
sozinha se roda a imagem inteira ou em tiles (acima de 768 px de lado passa pro modo
tiled com média de logits, que foi a conclusão da Parte 4).

## Testes

```
uv run pytest
```

Cobre o gerador sintético, as métricas de instância com caso montado na mão (match
perfeito dá mAP 1, um FN dá 0.5, um FP dá 2/3, dois objetos fundidos num blob só dá
menos de 0.5), os alvos da trilha A e a fusão de instâncias entre tiles da Parte 4.

O teste que mais vale a pena olhar é o do teto de oráculo em
`tests/test_targets_and_postprocess.py`: com a fronteira e a distância perfeitas o
watershed dá mAP acima de 0.70 no sintético, enquanto componentes conexos com máscara
semântica perfeita não passa de 0.11. Esse número é o argumento inteiro da Parte 2, e
dá pra checar sem treinar nada.

## Onde está cada parte

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

Não tem GPU nas nossas máquinas, então os treinos saíram em kernel do Kaggle com T4.
`notebooks/colab_setup.ipynb` faz a mesma coisa no Colab. Tudo roda em CPU também, só
demora: o baseline do DSB2018 leva algumas horas em vez de alguns minutos.
