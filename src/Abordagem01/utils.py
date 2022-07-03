# Funções matemáticas úteis para o desenvolvimento 
# do projeto que não encaixavam em outros arquivos:

import numpy as np

# Função para normalizar somatório de um array:
def normalizer(array):
    return np.array(array) / (np.sum(array))

# Ajusta valor para caber em determinado intervalo:
def adusting_range(num, low, high):
    if(num < low):
        return low
        
    elif(num > high):
        return high
    
    return num

