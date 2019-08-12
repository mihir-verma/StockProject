import datetime
import quandl
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style

#!
style.use('ggplot')
#--


#!
quandl.ApiConfig.api_key = 'qr5Q6WaDj1FA4p7m6one'

reliance_df = quandl.get('BSE/BOM500325')
pd.to_pickle(reliance_df, 'reliance_raw.pickle')
#--

#!
reliance_df = pd.read_pickle('reliance_raw.pickle')
#print(reliance_df.head())
#--


#!
#today = datetime.datetime(2019, 3, 8, 00, 00, 00)
#df_today = pd.DataFrame({'Close': [1266.65]}, index=[today])
#reliance_df = reliance_df.append(df_today, sort=True)
#--


#!
def get_close_val(stock_df):
    return stock_df.filter(items=['Close'])
#--

#!   
def bollinger_bands(stock_df):
    bol_upp = stock_df['Close'].rolling(20).mean() + ( 2 * stock_df['Close'].rolling(20).std() )
    bol_mid = stock_df['Close'].rolling(20).mean()
    bol_low = stock_df['Close'].rolling(20).mean() - ( 2 * stock_df['Close'].rolling(20).std() )
    return round(bol_upp, 3), round(bol_mid, 3), round(bol_low, 3)
#--

#!  
def macd_indicator(stock_df):
    macd_line = stock_df['Close'].ewm(span = 12).mean() - stock_df['Close'].ewm(span = 26).mean()
    macd_signal = macd_line.ewm(span = 9).mean()
    macd_histogram = macd_line - macd_signal
    return round(macd_line, 3), round(macd_signal, 3), round(macd_histogram, 3)
#--

#!    
def rsi_indicator(stock_df):
    delta = stock_df['Close'].diff()[1:]
    up, down = delta.copy(), delta.copy()
    
    up[up < 0] = 0
    down[down > 0] = 0
    
    roll_up = up.ewm(span=14).mean()
    roll_down = down.abs().ewm(span=14).mean()
    
    RS = roll_up/roll_down
    RSI = 100 - (100 / (1 + RS) )
    
    return round(RSI, 3)
#--

#!    
def visualize(stock_df):
    df_0 = stock_df[['Close','bol_upp','bol_mid','bol_low']]
    df_1 = stock_df[['macd_line','macd_signal','macd_histogram']]
    df_2 = stock_df[['rsi_val']]
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((2,2),(0,0), colspan=2)
    ax2 = plt.subplot2grid((2,2),(1,0), sharex=ax1)
    ax3 = plt.subplot2grid((2,2),(1,1), sharex=ax1)
    
    df_0.plot(ax = ax1)
    df_1.plot(ax = ax2)
    df_2.plot(ax = ax3)
    
    plt.show()
    
def getbuysell(stock_df):
    stock_df = stock_df.filter(items=['Close'])
    stock_df.reset_index(inplace=True)
    stock_df['Close'] = stock_df['Close'].diff()
    stock_df['Close'][stock_df['Close'] > 0] = 1
    stock_df['Close'][stock_df['Close'] < 0] = 0
    stock_df['Indicator'] = stock_df['Close'].shift(-1)
    stock_df.drop(labels=['Close'], axis=1, inplace=True)
    stock_df.to_csv('RelianceBuySell.csv')
    #print(stock_df)
#--    


#!    
reliance_df = get_close_val(reliance_df)

reliance_df['bol_upp'], reliance_df['bol_mid'], reliance_df['bol_low'] = bollinger_bands(reliance_df)
reliance_df['macd_line'], reliance_df['macd_signal'], reliance_df['macd_histogram'] = macd_indicator(reliance_df)
reliance_df['rsi_val'] = rsi_indicator(reliance_df)

#getbuysell(reliance_df)
#print(reliance_df.tail())

visualize(reliance_df)
pd.to_pickle(reliance_df, 'reliance_prepared.pickle')
reliance_df.to_csv('Reliance.csv')
#--