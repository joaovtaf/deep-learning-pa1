# AI_LOG

Usamos Claude (Claude Code no VS Code) durante o PA. Anotando aqui os episódios
principais, o que a ferramenta fez e o que a gente teve que decidir ou corrigir na mão.

## Decidir dataset e trilha da Parte 2

Perguntamos qual das duas opções de dataset dava menos trabalho de engenharia e qual
das três trilhas da Parte 2 tende a dar resultado melhor. A resposta foi opção A
(DSB2018) porque não precisa de conta e as máscaras já vêm uma por núcleo, sem COCO
pra parsear, e trilha A (fronteira + watershed) porque o pós-processamento é
determinístico e não depende de hiperparâmetro de clustering igual à trilha B, nem
de objeto convexo igual à trilha C. Concordamos com as duas e seguimos.

## Métricas de instância

Essa foi a parte que mais valeu discutir com a IA, porque o enunciado proíbe usar AP
de biblioteca. A implementação da matriz de IoU entre todas as instâncias previstas e
todas as verdadeiras saiu como um bincount de pares codificados em `pred * (n_gt+1) + gt`,
que evita o loop duplo sobre instâncias. A gente conferiu na mão que a conta bate:
tem teste com dois quadrados, um caso de IoU 8/16 conhecido, um FN, um FP, e o caso
dos dois objetos fundidos num blob só.

O ponto que a gente teve que decidir sozinho foi qual definição de precisão usar. O
slide 6 define AP como precisão média sobre as classes, que é semântica. Adotamos a
convenção do DSB2018, precisão no limiar t igual a TP / (TP + FP + FN), e AP da
imagem como média sobre os limiares 0.50 a 0.95. Deixamos isso escrito no docstring
de `metrics/instance.py` porque o PA pede que a regra esteja explícita.

## Gerador sintético da Parte 0

Primeira versão colocava as elipses em posição aleatória e quase nenhuma encostava,
o que não servia pro que a gente queria mostrar. Mudamos pra ancorar a maioria das
elipses novas ao lado de uma já colocada, com a distância entre centros perto da soma
dos raios. Depois disso apareceu outro bug nosso: elipse desenhada por cima podia
enterrar uma anterior e a imagem acabava com menos de 5 instâncias, fora da faixa que
o PA pede. Arrumamos contando quantas instâncias sobrevivem a cada passo em vez de
contar quantas foram desenhadas.

Sobrou um terceiro problema que só apareceu quando a gente foi medir o teto de
oráculo do watershed. Ele dava 0.74 em vez de perto de 1, mesmo recebendo a fronteira
e a distância perfeitas. Investigando imagem por imagem, o culpado eram instâncias de
6 a 8 pixels, restos de elipses quase totalmente enterradas por outra desenhada em
cima. Lasca de 6 pixels não é objeto, ninguém casa com ela, e ela puxava a métrica
inteira pra baixo. Colocamos um filtro de área mínima no gerador e o teto subiu pra
0.83. Isso é um bug do nosso dataset, não do método, e valeu a pena caçar porque o
número do teto de oráculo é o argumento central da Parte 2.

## Debug menor

`remove_small_objects` do skimage 0.26 deprecou o parâmetro `min_size` e ficava
cuspindo warning. Trocamos por uma função nossa de 8 linhas com bincount, que já
renumera os labels de quebra.

Outro: o dataset sintético devolve imagem 2D em float e o DSB2018 devolve RGB uint8, e
o código de tiling da Parte 4 assumia RGB. Em vez de espalhar `if`, centralizamos numa
função `as_rgb_uint8` em utils.

## Espessura da fronteira e o alpha

A IA sugeriu erodir cada instância separadamente pra gerar a classe fronteira, em vez
de erodir a máscara semântica inteira. A gente não tinha pensado nisso e é o detalhe
que faz a coisa funcionar: erodindo o foreground todo de uma vez, o contato entre dois
núcleos fica no meio da região e não vira fronteira nenhuma, então o watershed não
teria onde cortar. Escrevemos um teste específico pra isso
(`test_boundary_separates_touching_instances`).

A escolha da espessura e do alpha foi nossa, olhando a saída de `scripts/class_stats.py`,
que a gente escreveu justamente pra ver o trade-off: casca fina deixa a classe 2
minúscula demais e a rede ignora, casca grossa come o interior dos núcleos pequenos e
eles param de virar marcador no watershed (a coluna "perdidos" da tabela).

## Rodar em GPU

Nenhuma das nossas máquinas tem GPU NVIDIA, então os treinos foram parar em kernel do
Kaggle. A primeira tentativa caiu numa P100, que é sm_60, e o PyTorch da imagem do
Kaggle só cobre sm_70 pra cima, então CUDA aparecia como disponível mas nada rodava.
A solução foi o kernel checar `torch.cuda.get_device_capability()` no início e, se for
antiga, instalar um torch compatível antes de começar.

## O que a IA não decidiu

Escolha do encoder, o formato do split estratificado (a ideia de usar k-means sobre
estatística de cor como proxy de modalidade), a decisão de selecionar checkpoint por
mAP de validação em vez de loss, e quais dois eixos da Parte 3 atacar foram discutidos
mas a chamada final foi nossa. A leitura da Parte 5, que o problema do nosso encoder
não é campo receptivo pequeno e sim output stride grande demais pro tamanho do núcleo,
saiu de a gente olhar a tabela de campo receptivo do lado do histograma de tamanho e
perceber que os dois números não contavam a história que a gente esperava.
