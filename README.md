# Gabarito — Neuroevolução no Freeway

**UniSenac Pelotas | Fundamentos de Inteligência Artificial | Prof. Pablo De Chiaro Rosa**

Implementação de referência do trabalho prático. O objetivo é evoluir os **pesos de uma rede neural**, sem backpropagation, usando seleção, elitismo e mutação.

## Instalação

Requer Python 3.10+ e a ROM do Freeway fornecida pela instalação do `ale-py`.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
```

## Execução

```bash
python main.py                       # experimento B (20 indivíduos, 30 gerações)
python main.py --experimento todos   # executa A, B e C
python main.py --populacao 4 --geracoes 2  # teste rápido
python main.py --visualizar          # abre uma partida do melhor agente ao final
python main.py --acompanhar          # mostra cada indivíduo durante o treinamento
python main.py --experimento todos --acompanhar  # acompanha A, B e C em tempo real
python main.py --populacao 4 --geracoes 2 --acompanhar  # teste rápido com janela
```

### Google Colab / Jupyter

Instale as dependências na primeira célula:

```python
%pip install "gymnasium[atari]" ale-py numpy matplotlib
```

Em seguida, execute o arquivo na célula:

```python
%run main.py --populacao 4 --geracoes 2
```

Alternativamente, após importar `main`, é possível passar os parâmetros diretamente:

```python
from main import main
main(["--populacao", "4", "--geracoes", "2"])
```

O código ignora argumentos internos do kernel ao ser executado no Jupyter, mas mantém validação estrita no terminal. **Não use `--visualizar` nem `--acompanhar` no Colab**: `render_mode="human"` requer interface gráfica local.

A visualização requer ambiente gráfico. Com `--acompanhar`, o Freeway mostra **cada partida de cada indivíduo durante o treinamento**, e o console identifica geração e indivíduo. Isso é diferente de `--visualizar`, que mostra apenas o campeão no final. É possível combinar as opções. Uma execução completa pode demorar bastante: cada indivíduo joga uma partida inteira por geração, e a renderização ocorre em tempo real.

## Como funciona

- **Ambiente:** `ALE/Freeway-v5`, observação `ram` (128 entradas), 3 ações discretas.
- **Agente:** rede neural `128 → 16 (tanh) → 3 (argmax)`. Pesos e vieses formam o DNA.
- **Recompensa:** valor retornado pelo ambiente em cada passo. **Fitness:** soma das recompensas recebidas no episódio.
- **População:** 20 agentes com pesos inicialmente aleatórios.
- **Seleção:** melhores 50% pelo fitness; cada filho escolhe aleatoriamente um pai dentre eles.
- **Elitismo:** os 2 melhores são copiados sem mutação.
- **Reprodução:** cópia independente do pai, **sem crossover**.
- **Mutação:** cada peso ou viés tem uma probabilidade de alteração; a mudança é um ruído gaussiano com desvio-padrão igual à intensidade configurada.
- **Exploração:** ocorre na inicialização aleatória e nas mutações.
- **Convergência:** pode ser investigada pelas curvas de melhor fitness e fitness médio. Estabilização não garante solução ótima.

## Experimentos exigidos

| Experimento | Taxa de mutação | Intensidade |
|---|---:|---:|
| A | 1% | 0,05 |
| B | 10% | 0,20 |
| C | 40% | 0,50 |

Use `--experimento todos` para executar os três. A semente (`--seed`, padrão 42) é a mesma para as três configurações, permitindo comparação mais controlada. A avaliação utiliza um episódio por indivíduo; para conclusões estatísticas, recomenda-se repetir com várias sementes.

## Arquivos gerados

Cada experimento gera uma pasta `resultados/A`, `resultados/B` ou `resultados/C` com:

- `fitness.csv`: melhor, média e pior fitness por geração;
- `diagnostico.csv`: por geração e indivíduo, fitness, passos, percentual de NOOP/UP/DOWN, trocas de ação e dominância da ação mais usada;
- `fitness.png`: gráfico do melhor fitness e fitness médio;
- `melhor_agente.npz`: pesos do melhor indivíduo encontrado;
- `comparacao.csv`: recompensa do agente aleatório e do evoluído.

### Diagnóstico de comportamento

O console exibe, após cada geração, a distribuição média de ações dos agentes e quantos usam uma única ação em pelo menos **95%** dos passos. Esses agentes são classificados como **políticas repetitivas** para fins de diagnóstico, não necessariamente como políticas ruins.

O `diagnostico.csv` permite investigar cada indivíduo: `noop_pct`, `up_pct` e `down_pct` mostram a frequência das ações, `trocas_acao` indica mudanças de decisão e `acao_dominante_pct` mede a predominância de uma ação. Esses valores **não alteram o fitness, a seleção ou a mutação**; servem apenas para observação. A média no console é calculada entre indivíduos, não ponderada pelo número de passos.

**Interpretação:** a comparação aleatória é uma baseline simples, não prova que o evoluído seja melhor em geral. Uma população pode estagnar; comparar as três curvas e executar múltiplas sementes oferece evidência mais forte.

## Correspondência com a atividade

| Conceito / etapa | Implementação |
|---|---|
| Parte 1: rede e ambiente | `Agente`, `criar_ambiente` |
| Parte 2: interação e fitness | `executar_episodio`, `avaliar` |
| Parte 2: população e seleção | `criar_populacao`, `selecionar` |
| Parte 2: elitismo e mutação | `mutar`, `nova_geracao` |
| Parte 3: gerações, métricas e gráfico | `treinar`, `salvar_historico` |
| Comparação e visualização | `comparar`, `visualizar` |

> Gabarito didático: os resultados numéricos dependem da execução e não são presumidos neste documento.
