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
o que nao servia pro que a gente queria mostrar. Mudamos pra ancorar a maioria das
elipses novas ao lado de uma ja colocada, com a distancia entre centros perto da soma
dos raios. Depois disso apareceu outro bug nosso: elipse desenhada por cima podia
enterrar uma anterior e a imagem acabava com menos de 5 instancias, fora da faixa que
o PA pede. Arrumamos contando quantas instancias sobrevivem a cada passo em vez de
contar quantas foram desenhadas.

Sobrou um terceiro problema que so apareceu quando a gente foi medir o teto de
oraculo do watershed. Ele dava 0.74 em vez de perto de 1, mesmo recebendo a fronteira
e a distancia perfeitas. Investigando imagem por imagem, o culpado eram instancias de
6 a 8 pixels, restos de elipses quase totalmente enterradas por outra desenhada em
cima. Lasca de 6 pixel nao e objeto, ninguem casa com ela, e ela puxava a metrica
inteira pra baixo. Colocamos um filtro de area minima no gerador e o teto subiu pra
0.83. Isso e um bug do nosso dataset, nao do metodo, e valeu a pena caçar porque o
numero do teto de oraculo e o argumento central da Parte 2.

## Debug menor

`remove_small_objects` do skimage 0.26 deprecou o parametro `min_size` e ficava
cuspindo warning. Trocamos por uma funcao nossa de 8 linhas com bincount, que ja
renumera os labels de quebra.

Outro: o dataset sintetico devolve imagem 2D em float e o DSB2018 devolve RGB uint8, e
o codigo de tiling da Parte 4 assumia RGB. Em vez de espalhar `if`, centralizamos numa
funcao `as_rgb_uint8` em utils.

## Espessura da fronteira e o alpha

A IA sugeriu erodir cada instancia separadamente pra gerar a classe fronteira, em vez
de erodir a mascara semantica inteira. A gente nao tinha pensado nisso e e o detalhe
que faz a coisa funcionar: erodindo o foreground todo de uma vez, o contato entre dois
nucleos fica no meio da regiao e nao vira fronteira nenhuma, entao o watershed nao
teria onde cortar. Escrevemos um teste especifico pra isso
(`test_boundary_separates_touching_instances`).

A escolha da espessura e do alpha foi nossa, olhando a saida de `scripts/class_stats.py`,
que a gente escreveu justamente pra ver o trade-off: casca fina deixa a classe 2
minuscula demais e a rede ignora, casca grossa come o interior dos nucleos pequenos e
eles param de virar marcador no watershed (a coluna "perdidos" da tabela).

## Rodar em GPU

Nenhuma das nossas maquinas tem GPU NVIDIA, entao os treinos foram parar em kernel do
Kaggle. A primeira tentativa caiu numa P100, que e sm_60, e o PyTorch da imagem do
Kaggle so cobre sm_70 pra cima, entao CUDA aparecia como disponivel mas nada rodava.
A solucao foi o kernel checar `torch.cuda.get_device_capability()` no inicio e, se for
antiga, instalar um torch compativel antes de comecar.

## O que a IA nao decidiu

Escolha do encoder, o formato do split estratificado (a ideia de usar k-means sobre
estatistica de cor como proxy de modalidade), a decisao de selecionar checkpoint por
mAP de validacao em vez de loss, e quais dois eixos da Parte 3 atacar foram discutidos
mas a chamada final foi nossa. A leitura da Parte 5, que o problema do nosso encoder
nao e campo receptivo pequeno e sim output stride grande demais pro tamanho do nucleo,
saiu de a gente olhar a tabela de campo receptivo do lado do histograma de tamanho e
perceber que os dois numeros nao contavam a historia que a gente esperava.
