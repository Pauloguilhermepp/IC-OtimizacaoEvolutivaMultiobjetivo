# Classe para representar o algoritmo genético:
class Elemento:
    # Construtor:
    def __init__(self, array = None, fitness = None):
        # Variáveis relativas ao funcionamento do algoritmo:

        self.array = array # Array de pesos
        self.fitness = fitness # Fitness da solução

