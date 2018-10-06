# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('/Users/naveenkumar/Desktop/RollingxG.csv')

#Loading Data
Team_Data = dataset[dataset['Team'].isin(['Liverpool'])]

GW = pd.Series(Team_Data.iloc[5:, 0].values)
xG = pd.Series(Team_Data.iloc[:, 5].values).dropna()
xGA = pd.Series(Team_Data.iloc[:,6].values).dropna()

GW_Reshape = GW.values.reshape(-1, 1)

#Running Linear Regression Model

from sklearn.linear_model import LinearRegression
regressor_xG = LinearRegression()
regressor_xGA = LinearRegression()
regressor_xG.fit(GW_Reshape, xG_1)
regressor_xGA.fit(GW_Reshape, xGA_1)

#Plot the results
fig, ax = plt.subplots()
fig.set_size_inches(8,5)

xG_Plot = sns.scatterplot(x="Gameweek", y="Rolling xG", data=Team_Data, color="g", ax=ax).set_title(Team_1['Team'].iloc[0])
xG_Plot = sns.lineplot(x=GW, y=regressor_xG.predict(GW_Reshape), data=Team_1, color="g", ax=ax)

xGA_Plot = sns.scatterplot(x="Gameweek", y="Rolling xGA", data=Team_Data, color="r", ax=ax)
xGA_Plot = sns.lineplot(x=GW, y=regressor_xGA.predict(GW_Reshape), data=Team_1, color="r", ax=ax).set(ylabel="Rolling xG / xGA", ylim=(0,3), xlim=(5,39), xticklabels=[] )
