# Abordagem01: Ponderada - arquivo principal

import time
import numpy as np
import pandas as pd
from AG.algoritmo_genetico import *

# Definindo algumas variáveis globais:
pesos = [1, -1] # Pesos da função para se maximizar
returns = [] # Array para salvar retorno das ações
cov_matrix_annual = [] # Matriz da covariância anual

# Definindo a função fitness:
def fitness_func(solution):

    # Calculando retorno:
    portfolio_simple_annual_return = np.sum(returns.mean() * solution) * 252

    # Calculando volatilidade:
    port_variance = np.dot(solution.T, np.dot(cov_matrix_annual, solution))
    port_volatility = np.sqrt(port_variance)
    
    # Calculando a função a ser otimizada:
    fitness = pesos[0] * portfolio_simple_annual_return + pesos[1] * port_volatility
    return fitness


# Função para mostrar solução:
def show_ans(solution, solution_fitness):

    # Calculando retorno:
    portfolio_simple_annual_return = np.sum(returns.mean() * solution) * 252

    # Calculando volatilidade:
    port_variance = np.dot(solution.T, np.dot(cov_matrix_annual, solution))
    port_volatility = np.sqrt(port_variance)


    # Mostrando respostas:
    print("Parâmetros da melhor solução: {solution}".format(solution = solution))
    print("Fitness da melhor solução = {solution_fitness}".format(solution_fitness = solution_fitness))
    print("Retorno da melhor solução = {solution_return}".format(solution_return = portfolio_simple_annual_return))
    print("Volatilidade da melhor solução = {solution_vol}".format(solution_vol = port_volatility))


# Definindo atributos do algoritmo genético:
def genetic_algorithm(num_stocks):
    # Definindo parâmetros do AG:
    fitness_function = fitness_func # Função fitness

    num_generations = 50 # Número de gerações

    pop_size = 40 # Número de elementos da população
    
    num_genes = num_stocks # Número de variáveis

    gene_space = {'low': 0, 'high': 1} # Limites do valor das soluções

    verbose = False # Não mostrar dados relativos a resposta

    plot_graph = False # Não mostrar o gráfico

    mutation_type = "uniform" # Tipo de mutação que os genes vão ter

    mutation_param = 0.2 # Parâmetro relacionado a mutação

    
    # Criando o AG:
    ga_instance = AlgoritmoGenetico(pop_size = pop_size, 
                                num_generations = num_generations, 
                                fitness_func = fitness_function, 
                                num_genes = num_genes,
                                low = gene_space['low'],
                                high = gene_space['high'],
                                verbose = verbose,
                                plot_graph = plot_graph,
                                mutation_type = mutation_type,
                                mutation_param = mutation_param)

    return ga_instance


# Plotando histograma com todos os dados:
def plot_hist(fitness_values):
    plt.hist(fitness_values)
    plt.title("Histograma do fitness do melhor indivíduo", fontsize = 14)
    plt.xlabel("Fitness do melhor indivíduo", fontsize = 12)
    plt.ylabel("Número de ocorrências", fontsize = 12)
    plt.ticklabel_format(useOffset = False)
    plt.grid()
    plt.show()

# Função principal:
def main():
    global returns, cov_matrix_annual

    # Acessando dados:
    num_stocks = 10
    path = "Data/DataBase{ns}.pkl".format(ns = num_stocks)
    df = pd.read_pickle(path)
    
    # Calculando matrizes de retorno e covariância:
    returns = df.pct_change()
    cov_matrix_annual = returns.cov() * 252

    # Número de execuções do algoritmo:
    num_exe = 10
    fitness_values = []

    # Iniciando a contagem do tempo:
    start_time = time.time()

    # Execução do algoritmo em loop:
    for exe in np.arange(num_exe):
        # Criando instância do AG:
        ga_instance = genetic_algorithm(num_stocks)

        # Seed da simulação:
        # ga_instance.random_seed(exe)

        # Executando o AG:
        ga_instance.run()

        # Salvando resultado do AG:
        fitness_values.append(ga_instance.best_ans.fitness)

        # Mostrando detalhes da execução:
        # show_ans(ga_instance.best_ans.array, ga_instance.best_ans.fitness)

    # Parando a contagem do tempo:
    end_time = time.time()

    plot_hist(fitness_values)

    # Calculando duração total:
    time_duration = (end_time - start_time)
    average_time = time_duration / num_exe
    print("Tempo de execução total: {time_duration}".format(time_duration = time_duration))
    print("Tempo de execução médio: {average_time}".format(average_time = average_time))
    print(sum(fitness_values)/len(fitness_values))


if __name__ == "__main__":
    main()

