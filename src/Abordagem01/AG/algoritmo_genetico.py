import random
import numpy as np
from utils import *
from AG.elemento import *
import matplotlib.pyplot as plt


# Classe para representar o algoritmo genético:
class AlgoritmoGenetico:
    # Construtor:
    def __init__(self, pop_size, num_generations, fitness_func, num_genes, 
                low = -10, high = 10, r_mut = 1, k_tournament = 3, verbose = True,
                plot_graph = True, crossover_type = "two_points", mutation_type = "uniform",
                mutation_param = 1):
        

        # Definindo variáveis relativas ao funcionamento do algoritmo:

        self.pop_size = pop_size # Tamanho da população
        self.num_generations = num_generations # Número de gerações
        self.fitness_func = fitness_func # Função fitness
        self.num_genes = num_genes # Número de variáveis em uma solução
        
        self.low = low # Limite inferior de cada gene
        self.high = high # Limite superior de cada gene
        
        self.r_mut = r_mut # Frequência de mutações na população
        
        self.k_tournament = k_tournament # Número de confrontos no torneio
        
        self.verbose = verbose # Opção verbosa relativa a apresentação de dados do AG
        
        self.plot_graph = plot_graph # Opção para mostrar gráfico da evolução do melhor elemento
        
        self.crossover_type = crossover_type # Tipo de crossover do algoritmo
        self.mutation_type = mutation_type # Tipo de mutação que cada gene sofre
        self.mutation_param = mutation_param # Parâmetro relacionado a mutação
        
        self.current_best_fit = [] # Array para salvar melhor solução atual


        # Definindo métodos (tipo de mutação, crossover e etc) usados pelo AG:
        self.def_methods()


    # Método que define métodos usados pelo algoritmo:
    def def_methods(self):
        # Sobre o crossover:
        if(self.crossover_type == "one_point"):
            self.crossover = self.one_point_crossover
        elif(self.crossover_type == "two_points"):
            self.crossover = self.one_point_crossover

        # Sobre a mutação:
        if(self.mutation_type == "uniform"):
            self.mutation = self.uniform_mutation
        elif(self.mutation_type == "normal"):
            self.mutation = self.normal_mutation


    # Função para normalizar arrays de uma população do AG:
    def normalize_pop(self):
        for element in np.arange(self.pop_size):
            self.pop[element].array = normalizer(self.pop[element].array)
    
    
    # Função para "definir" aleatoriedade do AG:
    def random_seed(self, num):
        random.seed(num)
        np.random.seed(num)


    # Iniciando a população:
    def start_pop(self):
        self.pop = np.ndarray((self.pop_size,), dtype = np.object)
        for i in np.arange(self.pop_size):
            self.pop[i] =  Elemento(array = np.random.uniform(low=self.low, high=self.high, size=self.num_genes))


    # Avaliação da população:
    def eval_pop(self):
        for i in np.arange(self.pop_size):
            self.pop[i].fitness = self.fitness_func(self.pop[i].array)

            if(self.pop[i].fitness > self.best_ans.fitness):
                self.best_ans = self.pop[i]
            
        self.current_best_fit.append(self.best_ans.fitness)


    # Método do torneio par escolha dos progenitores:
    def tournament_selection(self):
        for _ in np.arange(self.pop_size):
            parent = random.randint(0, self.pop_size-1)

            for ix in random.sample(range(0, self.pop_size), self.k_tournament - 1):
                if(self.pop[ix].fitness > self.pop[parent].fitness):
                    parent = ix

        return self.pop[parent]


    # Escolha dos progenitores:
    def parents_selection(self):
        self.parents = np.ndarray((self.pop_size,), dtype = np.object)

        for i in np.arange(self.pop_size):
            self.parents[i] = self.tournament_selection()


    # Crossover entre dois progenitores com corte em um ponto:
    def one_point_crossover(self, id1, id2):
        pt = random.randint(1, self.num_genes-1)

        c1 = np.concatenate((self.parents[id1].array[:pt], self.parents[id2].array[pt:]), axis = None)
        c2 = np.concatenate((self.parents[id2].array[:pt], self.parents[id1].array[pt:]), axis = None)

        return [c1, c2]
    
    
    # Crossover entre dois progenitores com corte em dois pontos:
    def two_points_crossover(self, id1, id2):
        # pt1, pt2 = random.sample(range(0, self.num_genes), 2)

        pt1 = random.randint(1, self.num_genes-1)
        pt2 = random.randint(1, self.num_genes-1)

        pt1, pt2 = np.sort([pt1, pt2])

        c1 = np.concatenate((self.parents[id1].array[:pt1],
                             self.parents[id2].array[pt1:pt2],
                             self.parents[id1].array[pt2:]),
                             axis = None)

        c2 = np.concatenate((self.parents[id2].array[:pt1],
                             self.parents[id1].array[pt1:pt2],
                             self.parents[id2].array[pt2:]),
                             axis = None)
        
        return [c1, c2]


    # Criando nova população:
    def new_generation(self):
        for i in np.arange(0, self.pop_size, 2):
            c1, c2 =  self.crossover(i, i + 1)
            self.pop[i], self.pop[i+1] = Elemento(array = c1), Elemento(array = c2)


    # Função da mutação com distribuição uniforme:
    def uniform_mutation(self):
        for ix in np.arange(self.pop_size):
            if(random.random() < self.r_mut):
                gene = random.randint(0, self.num_genes - 1)
                self.pop[ix].array[gene] = np.random.uniform(self.pop[ix].array[gene] - self.mutation_param/2,
                                                        self.pop[ix].array[gene] + self.mutation_param/2)

                # Ajustando valor para caber no range máximo:
                self.pop[ix].array[gene] = adusting_range(self.pop[ix].array[gene], 
                                                            self.low, self.high)


    # Função da mutação com distribuição normal:
    def normal_mutation(self):
        for ix in np.arange(self.pop_size):
            if(random.random() < self.r_mut):
                gene = random.randint(0, self.num_genes - 1)
                self.pop[ix].array[gene] = np.random.normal(self.pop[ix].array[gene], 
                                                            self.mutation_param, 1)

                # Ajustando valor para caber no range máximo:
                self.pop[ix].array[gene] = adusting_range(self.pop[ix].array[gene], 
                                                            self.low, self.high)


    # Mostrando solução:
    def show_ans(self):
        print("Melhor indivíduo: {melhor_ind}".format(melhor_ind = self.best_ans.array))
        print("Fitness: {fitness}".format(fitness = self.best_ans.fitness))
    

    # Mostrando gráfico:
    def show_graph(self, title = "Gráfico do fitness do melhor indivíduo pelo número de gerações",
                xlabel = "Número de gerações", ylabel = "Fitness do melhor indivíduo"):

        plt.plot(range(0, self.num_generations + 1), self.current_best_fit)
        plt.title(title, fontsize = 14)
        plt.xlabel(xlabel, fontsize = 12)
        plt.ylabel(ylabel, fontsize = 12)
        plt.grid(b=True)
        plt.show()


    # Função para rodar o algoritmo genético:
    def run(self):
        # Iniciando a melhor resposta:
        self.best_ans = Elemento(array = 
            normalizer(np.random.uniform(low=self.low, high=self.high, size=self.num_genes)))
        self.best_ans.fitness = self.fitness_func(self.best_ans.array)

        # Iniciando a população:
        self.start_pop()

        # Normalizando população:
        self.normalize_pop()

        # Avaliando a população:
        self.eval_pop()


        for _ in np.arange(self.num_generations):
            # Escolha dos progenitores:
            self.parents_selection()

            # Nova geração:
            self.new_generation()

            # Mutação:
            self.mutation()

            # Normalizando população:
            self.normalize_pop()

            # Avaliação da população:
            self.eval_pop()


        # Mostrando dados relativos ao AG:
        if(self.verbose):
            self.show_ans()

        if(self.plot_graph):
            self.show_graph()

