# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Importing the dataset
dataset = pd.read_csv('/Users/naveenkumar/Desktop/RollingxG.csv')

#Set Number of iterations of Gradient Descent
Iterations = 10000

#Load Input Data
Team_Data = dataset[dataset['Team'].isin(['Manchester United'])]

GW = tf.placeholder(tf.float32, [None])
xG = tf.placeholder(tf.float32, [None])
xGA = tf.placeholder(tf.float32, [None])

GW = pd.Series(Team_Data.iloc[5:, 0].values).astype('float32')
xG = pd.Series(Team_Data.iloc[:, 5].values).astype('float32').dropna()
xGA = pd.Series(Team_Data.iloc[:,6].values).astype('float32').dropna()

#Set Weights and Biases
W1 = tf.Variable(np.random.randn())
B1 = tf.Variable(np.random.randn())

W2 = tf.Variable(np.random.randn())
B2 = tf.Variable(np.random.randn())

X = tf.placeholder("float")
Y = tf.placeholder("float")
Y_ = tf.placeholder("float")

#Run Linear Regression
Y_pred = tf.add(tf.multiply(GW, W1), B1)
cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (66)
optimizer = tf.train.GradientDescentOptimizer(0.003).minimize(cost)

Y_pred_xGA = tf.add(tf.multiply(X, W2), B2)
cost_xGA = tf.reduce_sum(tf.pow(Y_pred_xGA - Y_, 2)) / (66)
optimizer_xGA = tf.train.GradientDescentOptimizer(0.003).minimize(cost_xGA)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range (Iterations):
    sess.run(optimizer, feed_dict={X: GW, Y: xG})
    sess.run(optimizer_xGA, feed_dict={X: GW, Y_: xGA})

#Plot Results
fig, ax = plt.subplots()
fig.set_size_inches(8,5)

xG_Plot = sns.scatterplot(x="Gameweek", y="Rolling xG", data=Team_Data, color="g", ax=ax).set_title(Team_Data['Team'].iloc[0])
xG_Plot = sns.lineplot(x=GW, y=sess.run(W1) * GW + sess.run(B1), data=Team_Data, color="g", ax=ax)

xGA_Plot = sns.scatterplot(x="Gameweek", y="Rolling xGA", data=Team_Data, color="r", ax=ax)
xGA_Plot = sns.lineplot(x=GW, y=sess.run(W2) * GW + sess.run(B2), data=Team_Data, color="r", ax=ax).set(ylabel="Rolling xG / xGA", ylim=(0,3), xlim=(5,39), xticklabels=[])
