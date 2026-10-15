# Evolução de um agente inteligente no Atari — Breakout

Objetivo: Construir um agente evolutivo através do reinforcement learning.

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuração do Breakout

| Item | Valor |
|---|---|
| Ambiente | `ALE/Breakout-v5` (frameskip = 4) |
| Observação | RAM do Atari
| Rede | 128 entradas → 16 ocultos (tanh) → 3 saídas (argmax) |
| Ações | 0 = NOOP, 2 = RIGHT, 3 = LEFT (ação 1 "SHOOT" não mapeada)|
| Tamanho da população | 20 |
| Número de gerações | 30 |
| Pais selecionados | X |
| Elite | X |
| Taxa / intensidade de mutação | A: 1% / 0,05 — B: 10% / 0,20 — C: 40% / 0,50 |

## Perguntas da defesa

### 1. O que é o agente e qual estado ele recebe do ambiente?
É uma rede neural cujos pesos são W1, b1, W2, b2. <br>
O estado que ele recebe é a RAM do Atari (obs_type), um vetor de 128 bytes.

### 2. Qual a diferença entre ação, recompensa e fitness?
- Ação: é o movimento que o agente faz no jogo, através do método env.step()
- Recompensa: é o valor que o ambiente devolve de cada ação. No Breakout, é cada tijolo quebrado.
- Fitness: é o desempenho total de um indivíduo, basicamente a soma da recompensa naquela partida.

### 3. Como indivíduo, população e geração aparecem no código?
- Indivíduo: é uma classe Agente() instanciada pelo criar_populacao()
- População: A população faz parte de cada geração. No código a primeira população é chamada pelo evoluir(), que da início aos experimentos. Depois disso, cada nova geração recebe por parametro a população atual a fim de construir a próxima geração.
- Geração: A geração aparece no loop dentro de evoluir(). Após a primeira população toda jogar, a nova geração é chamada. É na nova_geracao() que os melhores individuos são selecionados e seus filhos, possivelmente mutados, com intuito de passar o melhor DNA para a geração seguinte e procurar encontrar melhores modificações. (Pra mim é onde fica a "regra de negócio" da evolução)

### 4. Como a seleção foi implementada e por que esse critério foi escolhido?
Os pais são selecionados de duas formas, pelo maior fitness e pelo pico de tempo em uma fase.
O maior fitness é foi escolhido pois é a melhor forma de avaliar o desempenho de um individuo, já o tempo condiz também com o desempenho no Breakout, pois quanto mais tempo vivo, mais ele rebate a bolinha e mais tijolos são quebrados, porém ao pegar a média de tempo, ficavam todos valores muito próximos. Por isso utilizei o max tempo vivo, a fim de dar chance a outros individuos que nao desempenharam tao bem no fitness, porém conseguiram jogar "vida" boa.

### 5. O que aconteceria se não existisse elitismo?
Sem o elitismo, os melhores genes poderiam se perder, e o desempenho das gerações seguintes poderia piorar.


### 6. Como taxa e intensidade de mutação afetam os descendentes?
- Taxa: A cada novo descendente, a chance de ele sofrer uma mutação é designada pela taxa.
- Intensidade: A intensidade depende do individuo sofrer mutação, neste caso, ela define o quanto os pesos da rede neural variam.

### 7. Onde ocorre exploração no algoritmo?
Nos parametros aleatórios do Agente(), em mutar(), selecionar() e nova_geracao().

### 8. Os resultados indicam convergência? Qual evidência sustenta a resposta?
Os resultados apresentam uma melhora na média, porém um platô no melhor desempenho.
O gráfico sustenta essa resposta.

### 9. O melhor agente superou o agente aleatório? Qual evidência mostra isso?
Sim. A média do agente aleatório é muito menor que a média dos experimentos A, B e C.

### 10. Qual mudança vocês fariam para tentar melhorar o resultado?
- Bônus por rebater a bola: o jogo só pontua quando um tijolo quebra, então um agente que rebate mas não acerta tijolos fica com o mesmo fitness de quem não faz nada.
- Penalidade por vida perdida: Se o individuo aprender a ficar vivo, consequentemente, vai conseguir uma melhor pontuação.