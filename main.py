"""Gabarito didático: neuroevolução no Atari Freeway (ALE)."""

import argparse
import copy
import csv
import sys
from pathlib import Path

import ale_py
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


gym.register_envs(ale_py)

ENV_ID = "ALE/Breakout-v5"
NUM_ENTRADAS = 128
NUM_OCULTOS = 16
NUM_ACOES = 3
ACOES = [0, 2, 3] ## NOOP, RIGHT, LEFT
RESULTADOS = Path("resultados")
EXPERIMENTOS = {"A": (0.01, 0.05), "B": (0.10, 0.20), "C": (0.40, 0.50)}


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
    return gym.make(ENV_ID, obs_type="ram", render_mode=render_mode)


def executar_episodio(ambiente, agente=None, seed=42, diagnosticar=False):
    """Executa um episódio e, opcionalmente, contabiliza as ações."""
    estado, _ = ambiente.reset(seed=seed)
    aleatorio = np.random.default_rng(seed) if agente is None else None
    recompensa_total = 0.0
    contagem_acoes = np.zeros(NUM_ACOES, dtype=int)
    trocas_acao = 0
    acao_anterior = None
    terminado = False
    vida_total = 5
    ambiente.step(1)
    frame_inicio = 0

    while not terminado:
        acao = agente.agir(estado) if agente else int(aleatorio.integers(NUM_ACOES))
        if diagnosticar:
            contagem_acoes[acao] += 1
            if acao_anterior is not None and acao != acao_anterior:
                trocas_acao += 1
            acao_anterior = acao
        estado, recompensa, terminated, truncated, info = ambiente.step(acao)
        vida_atual = info.get("lives", 0)
        freame_atual = info.get("frame_number", 0)
        if vida_total == vida_atual:
          print(info)
        else:
          vida_total = vida_atual
          duracao_vida = freame_atual - frame_inicio
          frame_inicio = freame_atual
          recompensa_total += (duracao_vida/100) ** 2
          ambiente.step(1)
        recompensa_total += recompensa
        terminado = terminated or truncated

    if not diagnosticar:
        return recompensa_total

    passos = int(contagem_acoes.sum())
    percentuais = 100 * contagem_acoes / max(1, passos)
    return recompensa_total, {
        "noop_pct": float(percentuais[0]),
        "right_pct": float(percentuais[1]),
        "left_pct": float(percentuais[2]),
        "trocas_acao": trocas_acao,
        "acao_dominante_pct": float(max(percentuais)),
    }


def avaliar(agente, ambiente, seed=42, diagnosticar=False):
    """Fitness = recompensa acumulada em um episódio."""
    return executar_episodio(ambiente, agente, seed, diagnosticar)


def criar_populacao(tamanho):
    return [Agente() for _ in range(tamanho)]


def selecionar(populacao):
    """Seleciona a metade dos agentes com maior fitness."""
    ordenados = sorted(populacao, key=lambda agente: agente.fitness, reverse=True)
    return ordenados[: max(1, len(ordenados) // 2)]


def mutar(agente, taxa, intensidade):
    """Altera cada parâmetro com probabilidade 'taxa'."""
    for nome in ("W1", "b1", "W2", "b2"):
        pesos = getattr(agente, nome)
        mascara = np.random.random(pesos.shape) < taxa
        pesos += mascara * np.random.normal(0, intensidade, pesos.shape)
    agente.fitness = 0.0
    return agente


def nova_geracao(populacao, num_elite, taxa, intensidade):
    pais = selecionar(populacao)
    elite = [copy.deepcopy(pai) for pai in pais[:num_elite]]
    filhos = []

    while len(elite) + len(filhos) < len(populacao):
        pai = pais[np.random.randint(len(pais))]
        filho = copy.deepcopy(pai)
        filhos.append(mutar(filho, taxa, intensidade))

    return elite + filhos


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


def treinar(tamanho, geracoes, elite, taxa, intensidade, pasta, seed, acompanhar=False):
    np.random.seed(seed)
    populacao = criar_populacao(tamanho)
    historico = []
    diagnosticos = []
    melhor_agente = None
    melhor_fitness = float("-inf")

    ambiente = criar_ambiente(render_mode="human" if acompanhar else None)
    try:
        for geracao in range(geracoes):
            for indice, agente in enumerate(populacao, start=1):
                if acompanhar:
                    print(f"  Geração {geracao + 1}/{geracoes} | "
                          f"Indivíduo {indice}/{tamanho}", flush=True)
                agente.fitness, metricas = avaliar(agente, ambiente, seed, diagnosticar=True)
                diagnosticos.append({
                    "geracao": geracao,
                    "individuo": indice,
                    "fitness": agente.fitness,
                    **metricas,
                })
          
            diagnosticos_geracao = diagnosticos[-tamanho:]
            repetitivos = sum(d["acao_dominante_pct"] >= 95 for d in diagnosticos_geracao)
            medias_acoes = {
                acao: float(np.mean([d[acao] for d in diagnosticos_geracao]))
                for acao in ("noop_pct", "left_pct", "right_pct")
            }
            valores = [agente.fitness for agente in populacao]
            campeao = max(populacao, key=lambda agente: agente.fitness)
            if campeao.fitness > melhor_fitness:
                melhor_fitness = campeao.fitness
                melhor_agente = copy.deepcopy(campeao)

            linha = {
                "geracao": geracao,
                "melhor": float(max(valores)),
                "media": float(np.mean(valores)),
                "pior": float(min(valores)),
            }
            historico.append(linha)
            print(f"Geração {geracao:03d} | Melhor: {linha['melhor']:.2f} | "
                  f"Média: {linha['media']:.2f} | Pior: {linha['pior']:.2f}")
            print(f"  Ações médias: NOOP {medias_acoes['noop_pct']:.1f}% | "
                  f"ESQUERDA {medias_acoes['left_pct']:.1f}% | "
                  f"DIREITA {medias_acoes['right_pct']:.1f}% | "
                  f"Políticas repetitivas (>=95% uma ação): {repetitivos}/{tamanho}")

            if geracao < geracoes - 1:
                populacao = nova_geracao(populacao, elite, taxa, intensidade)
    finally:
        ambiente.close()

    pasta.mkdir(parents=True, exist_ok=True)
    salvar_historico(historico, pasta)
    with (pasta / "diagnostico.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(diagnosticos[0]))
        escritor.writeheader()
        escritor.writerows(diagnosticos)
    np.savez(pasta / "melhor_agente.npz", W1=melhor_agente.W1, b1=melhor_agente.b1,
             W2=melhor_agente.W2, b2=melhor_agente.b2)
    return melhor_agente, melhor_fitness


def comparar(agente, seed):
    ambiente = criar_ambiente()
    try:
        aleatorio = executar_episodio(ambiente, seed=seed)
        evoluido = avaliar(agente, ambiente, seed=seed)
    finally:
        ambiente.close()
    print(f"Comparação | Aleatório: {aleatorio:.2f} | Evoluído: {evoluido:.2f}")
    return aleatorio, evoluido


def visualizar(agente, seed):
    ambiente = criar_ambiente(render_mode="human")
    try:
        executar_episodio(ambiente, agente, seed)
    finally:
        ambiente.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Neuroevolução no Freeway")
    parser.add_argument("--populacao", type=int, default=20)
    parser.add_argument("--geracoes", type=int, default=30)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--experimento", choices=["A", "B", "C", "todos"], default="B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualizar", action="store_true",
                        help="Exibe o melhor agente ao terminar o treinamento.")
    parser.add_argument("--acompanhar", action="store_true",
                        help="Exibe todas as partidas enquanto os agentes são avaliados.")
    # Colab/Jupyter inclui argumentos internos do kernel (ex.: -f arquivo.json).
    # No terminal, argumentos inválidos continuam produzindo erro.
    if argv is None and "ipykernel" in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args(argv)

    if args.populacao < 2 or args.geracoes < 1 or not (1 <= args.elite <= args.populacao // 2):
        parser.error("Use população >= 2, gerações >= 1 e elite entre 1 e metade da população.")

    nomes = EXPERIMENTOS if args.experimento == "todos" else [args.experimento]
    for nome in nomes:
        taxa, intensidade = EXPERIMENTOS[nome]
        pasta = RESULTADOS / nome
        print(f"\nExperimento {nome} | Mutação: {taxa:.0%} | Intensidade: {intensidade}")
        agente, _ = treinar(args.populacao, args.geracoes, args.elite,
                           taxa, intensidade, pasta, args.seed, args.acompanhar)
        aleatorio, evoluido = comparar(agente, args.seed)
        with (pasta / "comparacao.csv").open("w", newline="", encoding="utf-8") as arquivo:
            escritor = csv.writer(arquivo)
            escritor.writerow(["agente", "recompensa"])
            escritor.writerows([["aleatorio", aleatorio], ["evoluido", evoluido]])
        if args.visualizar:
            visualizar(agente, args.seed)


if __name__ == "__main__":
    main()
