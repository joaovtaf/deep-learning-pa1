

# CLAUDE.md

## Contexto

Repositório do Programming Assignment 1 de Deep Learning, curso de Ciência de Dados da FGV.
Trabalho de dois alunos do terceiro ano, feito em dupla. Não é código de empresa, não é
projeto profissional, é trabalho de faculdade. Tudo que for escrito aqui (código, comentários,
README, commits, relatórios) precisa parecer trabalho de aluno mesmo, não texto de assistente
de IA.

## Regras gerais de escrita

- Sem emoji. Nenhum, em lugar nenhum (código, commits, markdown, plots, nada).
- Sem travessão (—) nem em dash usado como pontuação. Se precisar separar uma ideia, usa
  vírgula, ponto, ou quebra em outra frase.
- Sem excesso de listas e headers em markdown. Relatório e README de aluno normalmente é
  parágrafo corrido, meio bagunçado, não um deck todo bonitinho com bullet point pra tudo.
- Evitar frases de assistente ("vale ressaltar", "é importante notar", "em suma",
  "furthermore", "let's dive into", "in conclusion"). Escreve do jeito que um aluno explicaria
  pro colega ou pro professor, direto, às vezes meio informal.
- Pode misturar português e inglês sem problema (nome de variável em inglês, comentário em
  português, esse tipo de mistura é normal em projeto de faculdade).
- Comentário no código é curto e só quando ajuda de verdade. Não precisa comentar toda função,
  não precisa docstring gigante. Comentário que só repete o que a linha já diz, não escreve.
- Não precisa tratamento de erro pra tudo. Se o código quebra numa situação estranha, tudo
  bem, não precisa try/except em tudo nem validação de input em toda função.
- Não precisa type hint em tudo se o resto do arquivo não usa.
- Nome de variável, título de gráfico etc não precisam estar perfeitos. Pequena
  inconsistência é normal e não precisa ser corrigida.
- README simples, sem seção "Features", "Overview", "Usage", "Contributing" empilhada só
  pra parecer completo. Só escreve o que tem conteúdo de verdade pra dizer.

## Commits

- Mensagem curta, direta, em minúsculo na maior parte das vezes.
- Sem dois pontos, sem prefixo tipo "feat:", "fix:", "chore:". Nada de conventional commits.
- Sem lista no corpo do commit, sem seção "Summary" ou "Changes".
- Só descreve o que mudou numa frase só, do jeito que alguém digitaria rápido sem pensar
  muito. Tipo "ajusta normalização dos dados", "corrige bug no loop de treino", "sobe
  primeira versão do modelo", "fix", "arruma plot", "esqueci de subir esse arquivo".
- Pode variar entre português e inglês de um commit pro outro, pode variar maiúscula/
  minúscula, isso é normal e não precisa ficar consistente.

## Notebooks e relatórios

- Voz natural, meio informal, não texto revisado e polido demais.
- Não precisa introdução/desenvolvimento/conclusão bem marcados igual redação. Pode ir
  direto pro que interessa.
- Análise pode ser mais curta e direta do que um relatório "completo" de livro-texto.

3. Regras de engenharia
Permitido: PyTorch, encoders pré-treinados em ImageNet (VGG/ResNet/Xception/MobileNet), albumentations,
scipy.ndimage, skimage.segmentation, sklearn.cluster.
Proibido:
•torchvision.models.detection, detectron2, mmdetection, ultralytics, SAM, Cellpose, StarDist,
ou qualquer modelo pronto de segmentação de instâncias;
•métricas de AP de instância prontas de biblioteca — vocês implementam o matching;
•clonar uma solução pronta do DSB2018. Se usarem ideia de repo ou paper, citem e reescrevam.