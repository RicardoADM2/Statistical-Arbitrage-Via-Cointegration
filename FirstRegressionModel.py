import yfinance as yf 
import numpy as np 
import statsmodels.api as sm 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stm

data=yf.download(["GLD","SLV"], start='2010-01-01', auto_adjust=True)["Close"] 
#Descarga de precios 
logdata=np.log(data) #Convierto los precios a logprices para medir el spread 
#Aqui empiezo el modelo de regresion OLS 
X=sm.add_constant(logdata["SLV"])
#logdata es un dataframe, entonces tomo como constante la columna GLD 
model=sm.OLS(logdata["GLD"], X).fit()
hedge_ratio=model.params["SLV"]
print(hedge_ratio)

spread = logdata["GLD"] - hedge_ratio * logdata["SLV"]
spread.plot(figsize=(12,6), title="Spread")
plt.show()

#Teniendo un hedge ratio de around 0.9 para el logprice, entendemos que por cada 1% que se 
#mueve SLV, GLD se mueve un 0.9%

#Ahora hacemos el test de adf

adftest=stm.adfuller(spread)
print(adftest) # p-value de 0.0432 por lo que es estacionario con un 95% de confianza 

# Ahora necesitamos hacer el half-life test para determinar el tiemp de mean-reversion
y=spread.diff().dropna() # calcula la diferencia del spread 
#dropna() es para eliminar los valores nulos (en este es el primero ya que no hay cambio)
z=sm.add_constant(spread.shift(1).dropna()) #mismo proceso de constante pero con el valor nuevo 
hfmodel=sm.OLS(y, z).fit() # OLS con el spread y su diferencia 
hfhedge_ratio=hfmodel.params[z.columns[1]]
print("hf:",hfhedge_ratio)


half_life = -np.log(2) /hfhedge_ratio # calculo del half-life con la formula 
# es la misma formula que se usa para calcular radioactive decay 
print("Half-life of mean reversion: ", half_life)

#Ahora vamos a usar un rolling window para calcular el rolling z-score 
#voy a usar un window de 120 dias (half-life of mean reversion) 

rollingmean=spread.rolling(window=200).mean() #rolling mean del spread
rollingstd=spread.rolling(window=200).std()  #STANDARD DEVIATION DEL SPREAD EN EL WINDOW 

zscore=(spread-rollingmean)/rollingstd # formula ggez
plot=zscore.plot(figsize=(12,6), title="Z-SCORE OF SPREAD") 
plt.axhline(y=-1.5, color='r', linestyle=':', label='Long spread') 
plt.axhline(y=1.5, color='g', linestyle=':', label='Short spread') 
plt.axhline(y=0, color='g', linestyle=':', label='Exit') 
plt.legend()
plt.show()
print("z-score",zscore)

#Una vez tengo el z-score, podria hacer el backtest de la estrategia. Sin embargo, prefiero 
#hacer una division en periodos para verificar la estacionaridad el spread y la relacion de 
#cointegracion 

#En este caso, al tener datos desde 2015, la division sera en periodos de 2 trading years
#(500 days) 

from statsmodels.regression.rolling import RollingOLS 

#para usar el window , uso una funcion ya existente rollingols 
model=RollingOLS(logdata["GLD"], X, window=500).fit()
rollinghedge_ratio=model.params["SLV"]
print("rolling hedge ratio:", rollinghedge_ratio)

rollingspread = logdata["GLD"] - rollinghedge_ratio * logdata["SLV"]
rollingspread.plot(figsize=(12,6), title="Rolling Spread")
plt.show()
rollingzscore=(rollingspread-rollingmean)/rollingstd

#el spread es mucho mayor de lo que esperaba, por lo que voy a volvr a ahcer el adf test 
#sobre el rolling spread 

rollingadftest=stm.adfuller(rollingspread.dropna())
print("rolling adf test:" ,rollingadftest) 

# We find a p-value of 0.0731, so non-stationarity is only confimred up to 10% confidence. 
# the stationarity confidence is lower through a rolling window than in a long window, 
# for this reason, it merits a kalman filter application to study the relation across time,
# as it is giving signs of time-dependent cointegration

#In order to compare a kalmam filter implementation with a simple implementation 
#of a cointegration based strategy, I will implement a backtest in which a z values of 
#-1.5 and 1.5 are used to enter a long or short position, respectively,
#  and a z value of 0.5 is used to exit the positiion.

#as we are doing pairs trading, the split of capital allocated to each asset per movement 
#will be determined by the hedge ratio, working towards a beta-neutral approach 
starting_capital=100000 #initial capital
zthreshold=1.5 #absolute value of the z-score at which we enter a position 
zexit=0.5 #abs value at which we liquidate positions 
fee_rate=0.001
positions=[]
notional=0.1*starting_capital 
position=0
df = pd.DataFrame({
    'GLD': data['GLD'],
    'SLV': data['SLV'],
    'spread': spread,
    'zscore': zscore,
})


for i in range(0, len(rollingzscore)): 
    if rollingzscore.iloc[i]>zthreshold and position==0:
       position=-1 
    elif rollingzscore.iloc[i]<-zthreshold and position==0:
       position=1 
    elif -zexit<rollingzscore.iloc[i]<zexit and position!=0:
       position=0
    positions.append(position)

df['position'] = positions
df['executed_position'] = df['position'].shift(1).fillna(0)

# This sets up the position taken every trading day,
# avoiding look-ahead bias by shifting our used price by one 

spread_ret = np.log(df['GLD']).diff() - rollinghedge_ratio * np.log(df['SLV']).diff()
df['strategy_ret'] = df['executed_position'] * spread_ret
df['trade'] = df['executed_position'].diff().fillna(0) != 0
df['fee'] = abs(df['executed_position'].diff().fillna(0)) * notional * fee_rate
df['net_ret'] = df['strategy_ret'] * notional - df['fee']
df['equity'] = starting_capital + df['net_ret'].cumsum()

Resulting_capital=df['equity'].iloc[-1]
finalbalance = (df['equity'].iloc[-1] / starting_capital) - 1
Sharpe_ratio = df['net_ret'].mean() / df['net_ret'].std() * np.sqrt(252)
Max_drawdown = (df['equity'] / df['equity'].cummax() - 1).min()

print("Resulting capital is:", Resulting_capital)
print("final balance is:",finalbalance)
print("Sharpe ratio:",Sharpe_ratio)
print("Max Drawdown:", Max_drawdown)


fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

df["zscore"].plot(ax=axes[0], title="Rolling Z-Score (200-day)", color="blue")
axes[0].axhline(y=zthreshold, color="r", linestyle="--")
axes[0].axhline(y=-zthreshold, color="g", linestyle="--")
axes[0].axhline(y=0, color="black", linestyle=":")
axes[0].grid(True)

df["equity"].plot(ax=axes[1], title="Portfolio Equity ($)", color="green")
axes[1].grid(True)
plt.tight_layout()
plt.show()

# So as it sits, not really jane street worthy with a final balance of 
# -0.041 and a sharpe ratio of -0.212. However, this is a simple implementation 
# whichi should improve using a kalman filter to make better (as well as finding a proper z
# score for entry and exit)


    
