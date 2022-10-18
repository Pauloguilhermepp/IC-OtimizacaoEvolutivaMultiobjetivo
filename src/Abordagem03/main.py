import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem

# Criando a classe que representa nosso problema de otimização
class Port_opt(Problem):

    def __init__(self, num_stocks):
        super().__init__(n_var=num_stocks,
                         n_obj=2,
                         n_constr=1,
                         xl=0.0,
                         xu=1.0)

    def _evaluate(self, population, out, *args, **kwargs):
        array = []
        sums = []

        for element in population:
            f1 = return_func(element)
            f2 = volatility_func(element)
            array.append((f1, f2))
            sums.append(abs(1 - np.sum(element))-0.01)

        out["F"] = np.array(array)
        out["G"] = np.array(sums)


# Definindo algumas variáveis globais:
returns = [] # Array para salvar retorno das ações
cov_matrix_annual = [] # Matriz da covariância anual

# Calculando retorno:
def return_func(solution):
    portfolio_simple_annual_return = np.sum(returns.mean() * solution) * 252

    return -portfolio_simple_annual_return

# Calculando volatilidade:
def volatility_func(solution):
    port_variance = np.dot(solution.T,np.dot(cov_matrix_annual, solution))
    port_volatility = np.sqrt(port_variance)

    return port_volatility

# Definindo atributos do algoritmo genético:
def genetic_algorithm(problem):
    algorithm = NSGA2(pop_size=200)

    res = minimize(problem=problem,
                algorithm=algorithm,
                termination=('n_gen', 100),
                seed=5,
                verbose=False)

    return res

# Mostrando dados obtidos pelo algoritmo genético:
def show_graph(res):
    var = []
    returns = []
    
    for i in range(0, len(res.F)):
        returns.append(-res.F[i][0])
        var.append(res.F[i][1])
    
    print(res.F)

    plt.scatter(var, returns, facecolor="none", edgecolor="red", color="black", alpha=0.7)
    plt.xlabel("Variance")
    plt.ylabel("Return")
    plt.title("Graph of dominant solutions of return and variance")
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

    # Rodando NSGA-II:
    problem = Port_opt(num_stocks)
    res = genetic_algorithm(problem)
    show_graph(res)

if __name__ == "__main__":
    main()

