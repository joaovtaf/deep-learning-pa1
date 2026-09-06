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
imagens. O split treino/val/teste e estratificado por modalidade: como o DSB2018
nao tem essa coluna, a gente descreve cada imagem com estatistica de cor e roda um
k-means pra separar fluorescencia, brightfield e histologia, e depois divide dentro
de cada cluster. Ta em `src/pa1/data/splits.py`.

O dataset sintetico da Parte 0 nao precisa baixar nada, ele gera na hora a partir
de uma seed por indice.

## Treinar

```
uv run python scripts/train.py --config configs/dsb2018_baseline.yaml
```

Pro teste unitario sintetico da Parte 0, que roda em menos de 5 minutos:

```
uv run python scripts/train.py --config configs/synthetic_baseline.yaml
```

O checkpoint salvo e o de melhor mAP de validacao, nao o de menor loss. Isso importa
porque da pra ter Dice quase 1 e mAP de instancia pessimo, que e justamente o que a
Parte 1 quer mostrar.

## Avaliar

```
uv run python scripts/evaluate.py --config configs/dsb2018_baseline.yaml --checkpoint runs/dsb2018_baseline/best.pt
```

Imprime IoU e Dice semanticos, mAP@[.50:.95], AP@.50 e erro absoluto medio de
contagem, e salva em `runs/<nome>/eval_test/` um metrics.json com a tabela por
imagem, o grafico de mAP e erro de contagem contra densidade de objetos (item 5 da
Parte 1) e alguns paineis qualitativos.

O matching e implementado por nos em `src/pa1/metrics/instance.py`, guloso por IoU
decrescente por padrao. Da pra trocar por Hungarian com `--matching hungarian`.
Precisao num limiar e TP / (TP + FP + FN), que e a convencao do DSB2018, e o AP da
imagem e a media sobre os limiares de 0.50 a 0.95 de 0.05 em 0.05.

## Testes

```
uv run pytest
```

Cobre o gerador sintetico e as metricas de instancia, com caso montado na mao
(match perfeito da mAP 1, um FN da 0.5, um FP da 2/3, dois objetos fundidos num
blob so da menos de 0.5).

## Onde esta cada parte

Parte 0 ta em `src/pa1/data/synthetic.py` mais o config `synthetic_baseline.yaml`.
Parte 1 e o `dsb2018_baseline.yaml` mais `postprocess/naive.py` (limiar + componentes
conexos) e `metrics/instance.py`. Partes 2 a 6 ainda nao entraram, a trilha escolhida
pra Parte 2 e a A (fronteira + watershed) e o `postprocess/watershed.py` ja ta com a
assinatura reservada.
