# Import the python libraries
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Função para salvar os dados:
def save_data(path, assets, stock_start_date, stock_end_date):

    # Create a dataframe to store the adjusted close price of the stocks:
    df = pd.DataFrame()

    # Store the adjusted close price of the sock into the df
    for stock in assets:
        df[stock] = yf.download(stock, 
                            start=stock_start_date, 
                            end=stock_end_date, 
                            progress=False)['Adj Close']


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


if __name__ == "__main__":
    path = "Data/DataBase5.pkl"
    assets = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']
    '''
    assets = ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 
        'AMZN', 'TSLA', 'FB', 'NVDA', 
        'QCOM', 'INTC', 'NFLX', 'PYPL',
        'AMD', 'HON', 'SHOP', 'SBUX',
        'SNAP', 'F', 'SQ', 'SPOT', 
        'TWTR', 'CTRA', 'PINS']
    '''
    save_data(path, assets, '2013-01-01', '2020-03-16')