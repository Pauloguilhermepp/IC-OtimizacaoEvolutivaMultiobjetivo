# Import the python libraries
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web

# Função para salvar os dados:
def save_data(path, assets, stockStartDate, stockEndDate):

    # Create a dataframe to store the adjusted close price of the stocks:
    df = pd.DataFrame()

    # Store the adjusted close price of the sock into the df
    for stock in assets:
        df[stock] = web.DataReader(stock,data_source = 'yahoo', 
        start=stockStartDate,end=stockEndDate)['Adj Close']

    # Salvando dados:
    df.to_pickle(path)


# Gráfico das ações do dataframe:
def show_graph(df):
    title  = 'Preço ajustado em função do tempo'
    xlabel = 'Data'
    ylabel = 'Preço ajustado em dolar ($)'
    
    # Create and plot the graph
    for c in df.columns.values:
        plt.plot(df[c], label=c)

    plt.title(title, fontsize = 14)
    plt.xlabel(xlabel,fontsize = 12)
    plt.ylabel(ylabel, fontsize = 12)
    plt.legend(df.columns.values, loc = 'upper left')
    plt.grid()
    plt.show()

