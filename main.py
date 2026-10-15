import argparse
import copy
import csv
import os
import random
from pathlib import Path

import ale_py
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


gym.register_envs(ale_py)

ENV_ID = "ALE/Breakout-v5"
TAMANHO_POPULACAO = 20
NUM_ENTRADAS = 128
NUM_ELITE = 2
NUM_GERACOES = 30
NUM_OCULTOS = 16
NUM_ACOES = 3
NUM_PAIS = 10
ACOES = [0, 2, 3] ## NOOP, RIGHT, LEFT
RESULTADOS = Path("resultados")
EXPERIMENTOS = {"A": (0.01, 0.05), "B": (0.10, 0.20), "C": (0.40, 0.50)}
CORES = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}


class Agente:
    """Rede 128 -> 16 -> 3; os pesos e vieses são o DNA."""

    def __init__(self):
        self.W1 = np.random.randn(NUM_ENTRADAS, NUM_OCULTOS)
        self.b1 = np.random.randn(NUM_OCULTOS)
        self.W2 = np.random.randn(NUM_OCULTOS, NUM_ACOES)
        self.b2 = np.random.randn(NUM_ACOES)
        self.fitness = 0.0
    def agir(self, estado):
        estado = estado.astype(np.float32) / 255.0
        oculto = np.tanh(estado @ self.W1 + self.b1)
        saida = oculto @ self.W2 + self.b2
        valor = int(np.argmax(saida))
        return valor
    
def criar_ambiente(render_mode=None):
    return gym.make(
        ENV_ID,
        obs_type="ram",
        render_mode=render_mode,
        frameskip=4,
    )
    
class AgenteAleatorio:
    def agir(self):
        return random.randrange(NUM_ACOES)



def avaliar(agente, render_mode=None):
    env = criar_ambiente(render_mode)
    state, info = env.reset()
    state, reward, terminated, truncated, info = env.step(1)

    recompensa_total = reward
    vidas = info["lives"]
    inicio_vida = info["episode_frame_number"]

    while not (terminated or truncated):
        action = agente.agir(state)
        state, reward, terminated, truncated, info = env.step(ACOES[action])
        recompensa_total += reward

        if info["lives"] < vidas:
            vidas = info["lives"]
            tempo_vivo = info["episode_frame_number"] - inicio_vida
            recompensa_total += (tempo_vivo / 100) ** 2
            inicio_vida = info["episode_frame_number"]

            if not (terminated or truncated):
                state, reward, terminated, truncated, info = env.step(1)
                recompensa_total += reward
    env.close()
    return recompensa_total


def avaliar_varias_vezes(agente, episodios):
    return [avaliar(agente) for _ in range(episodios)]


def criar_populacao(tamanho):
    return [Agente() for _ in range(tamanho)]


def selecionar(populacao):
    """Seleciona a metade dos agentes com maior fitness."""
    return sorted(populacao, key=lambda x: x.fitness, reverse=True)[:NUM_PAIS]


def mutar(agente, taxa, intensidade):
    """Altera cada parâmetro com probabilidade 'taxa'."""
    for nome in ("W1", "b1", "W2", "b2"):
        pesos = getattr(agente, nome)
        mascara = np.random.random(pesos.shape) < taxa
        pesos += mascara * np.random.normal(0, intensidade, pesos.shape)
    agente.fitness = 0.0
    return agente


def nova_geracao(populacao):
    pais = selecionar(populacao)
    nova_geracao = []
    for pai in pais[:NUM_ELITE]:
        nova_geracao.append(copy.deepcopy(pai))

    weights = [pai.fitness for pai in pais]
    if sum(weights) == 0:
        weights = None
    while len(nova_geracao) < TAMANHO_POPULACAO:
        pai = random.choices(pais, weights=weights)[0]
        filho = copy.deepcopy(pai)
        mutar(filho)
        nova_geracao.append(filho)
    return nova_geracao


def salvar_historico(historico, pasta):
    with (pasta / "fitness.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=["geracao", "melhor", "media", "pior"])
        escritor.writeheader()
        escritor.writerows(historico)

    geracoes = [linha["geracao"] for linha in historico]
    plt.figure(figsize=(8, 4))
    plt.plot(geracoes, [linha["melhor"] for linha in historico], label="Melhor")
    plt.plot(geracoes, [linha["media"] for linha in historico], label="Média")
    plt.xlabel("Geração")
    plt.ylabel("Fitness (recompensa acumulada)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pasta / "fitness.png", dpi=150)
    plt.close()

def evoluir(nome, taxa, intensidade):
    global TAXA_MUTACAO, INTENSIDADE_MUTACAO
    TAXA_MUTACAO = taxa
    INTENSIDADE_MUTACAO = intensidade
    print(f"\n=== Experimento {nome}: taxa {taxa:.0%} | intensidade {intensidade:.2f} ===")

    populacao = criar_populacao()
    historico = []
    melhor_agente = None

    for geracao in range(NUM_GERACOES):

        for agente in populacao:
            agente.fitness = avaliar(agente)
            if melhor_agente is None or agente.fitness > melhor_agente.fitness:
                melhor_agente = copy.deepcopy(agente)

        valores = [agente.fitness for agente in populacao]

        melhor = max(valores)
        media = float(np.mean(valores))
        pior = min(valores)

        historico.append({
            "experimento": nome,
            "taxa": taxa,
            "intensidade": intensidade,
            "geracao": geracao,
            "melhor": melhor,
            "media": media,
            "pior": pior,
        })

        print(
            f"Geração {geracao:03d} | "
            f"Melhor: {melhor:.2f} | "
            f"Média: {media:.2f} | "
            f"Pior: {pior:.2f}"
        )

        populacao = nova_geracao(populacao)

    return historico, melhor_agente


def salvar_csv(caminho, linhas):
    with open(caminho, "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=linhas[0].keys())
        escritor.writeheader()
        escritor.writerows(linhas)


def salvar_grafico(caminho, historicos):
    plt.figure(figsize=(9, 5))
    for nome, historico in historicos.items():
        cor = CORES.get(nome, "#2a78d6")
        geracoes = [h["geracao"] for h in historico]
        plt.plot(geracoes, [h["melhor"] for h in historico], color=cor,
                 linewidth=2, label=f"{nome} - melhor")
        plt.plot(geracoes, [h["media"] for h in historico], color=cor,
                 linewidth=2, linestyle="--", label=f"{nome} - média")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.title("Evolução do fitness - Breakout")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho)
    plt.close()


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Neuroevolução no Breakout")
    parser.add_argument("--experimento", choices=["A", "B", "C", "todos"], default="todos",
                        help="qual experimento rodar (padrão: todos)")
    parser.add_argument("--geracoes", type=int, default=NUM_GERACOES)
    parser.add_argument("--populacao", type=int, default=TAMANHO_POPULACAO)
    parser.add_argument("--render", action="store_true",
                        help="mostrar o melhor agente jogando ao final")
    return parser.parse_args()


def main():
    global NUM_GERACOES, TAMANHO_POPULACAO
    args = ler_argumentos()
    NUM_GERACOES = args.geracoes
    TAMANHO_POPULACAO = args.populacao
    nomes = list(EXPERIMENTOS) if args.experimento == "todos" else [args.experimento]
    os.makedirs("resultados", exist_ok=True)

    aleatorio = avaliar_varias_vezes(AgenteAleatorio(), args.episodios_teste)
    print(f"Agente aleatório: média {np.mean(aleatorio):.2f} | máximo {max(aleatorio):.2f}")
    comparacao = [{
        "agente": "aleatorio",
        "media": float(np.mean(aleatorio)),
        "maximo": max(aleatorio),
        "minimo": min(aleatorio),
    }]

    historicos = {}
    melhores = {}
    for nome in nomes:
        taxa, intensidade = EXPERIMENTOS[nome]
        historicos[nome], melhores[nome] = evoluir(nome, taxa, intensidade)

        teste = avaliar_varias_vezes(melhores[nome], args.episodios_teste)
        print(f"Melhor agente do experimento {nome}: média {np.mean(teste):.2f} "
              f"em {args.episodios_teste} episódios")
        comparacao.append({
            "agente": f"melhor_{nome}",
            "media": float(np.mean(teste)),
            "maximo": max(teste),
            "minimo": min(teste),
        })

    sufixo = "" if args.experimento == "todos" else f"_{args.experimento}"
    salvar_csv(f"resultados/fitness{sufixo}.csv", [h for n in nomes for h in historicos[n]])
    salvar_csv(f"resultados/comparacao{sufixo}.csv", comparacao)
    salvar_grafico(f"resultados/fitness{sufixo}.png", historicos)

    if args.render:
        nome = max(melhores, key=lambda n: melhores[n].fitness)
        print(f"Renderizando o melhor agente (experimento {nome})")
        print("Fitness ao jogar de novo:", avaliar(melhores[nome], render_mode="human"))


if __name__ == "__main__":
    main()
