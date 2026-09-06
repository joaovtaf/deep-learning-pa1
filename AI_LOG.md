# AI_LOG

Usamos Claude (Claude Code no VS Code) durante o PA. Anotando aqui os episodios
principais, o que a ferramenta fez e o que a gente teve que decidir ou corrigir na mao.

## Decidir dataset e trilha da Parte 2

Perguntamos qual das duas opcoes de dataset dava menos trabalho de engenharia e qual
das tres trilhas da Parte 2 tende a dar resultado melhor. A resposta foi opcao A
(DSB2018) porque nao precisa de conta e as mascaras ja vem uma por nucleo, sem COCO
pra parsear, e trilha A (fronteira + watershed) porque o pos-processamento e
deterministico e nao depende de hiperparametro de clustering igual a trilha B, nem
de objeto convexo igual a trilha C. Concordamos com as duas e seguimos.

## Metricas de instancia

Essa foi a parte que mais valeu discutir com a IA, porque o enunciado proibe usar AP
de biblioteca. A implementacao da matriz de IoU entre todas as instancias previstas e
todas as verdadeiras saiu como um bincount de pares codificados em `pred * (n_gt+1) + gt`,
que evita o loop duplo sobre instancias. A gente conferiu na mao que a conta bate:
tem teste com dois quadrados, um caso de IoU 8/16 conhecido, um FN, um FP, e o caso
dos dois objetos fundidos num blob so.

O ponto que a gente teve que decidir sozinho foi qual definicao de precisao usar. O
slide 6 define AP como precisao media sobre as classes, que e semantica. Adotamos a
convencao do DSB2018, precisao no limiar t igual a TP / (TP + FP + FN), e AP da
imagem como media sobre os limiares 0.50 a 0.95. Deixamos isso escrito no docstring
de `metrics/instance.py` porque o PA pede que a regra esteja explicita.

## Gerador sintetico da Parte 0

Primeira versao colocava as elipses em posicao aleatoria e quase nenhuma encostava,
o que nao servia pro que a gente queria mostrar. Mudamos pra ancorar 65% das elipses
novas ao lado de uma ja colocada, com a distancia entre centros perto da soma dos
raios. Depois disso apareceu outro bug nosso: elipse desenhada por cima podia enterrar
uma anterior e a imagem acabava com menos de 5 instancias, fora da faixa que o PA pede.
Arrumamos contando quantas instancias sobrevivem a cada passo em vez de contar quantas
foram desenhadas.

## Debug menor

`remove_small_objects` do skimage 0.26 deprecou o parametro `min_size` e ficava
cuspindo warning. Trocamos por uma funcao nossa de 8 linhas com bincount, que ja
renumera os labels de quebra.

## O que a IA nao decidiu

Escolha do encoder, o formato do split estratificado (a ideia de usar k-means sobre
estatistica de cor como proxy de modalidade), e a decisao de selecionar checkpoint por
mAP de validacao em vez de loss foram discutidas mas a chamada final foi nossa.
