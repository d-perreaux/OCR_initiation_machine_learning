import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/age_vs_poids_vs_taille_vs_sexe.csv')

print(df.head())

print(df.describe())

X = df[['sexe', 'age']]
y = df.poids

sns.scatterplot(x='taille', y='poids', data=df, hue="sexe")
plt.savefig('scatter.png')

sns.lmplot(x='taille', y='poids', data=df, hue="sexe")
plt.savefig('linear_model.png')

sns.pairplot(df, kind='reg')
plt.savefig('pairplot.png')

reg = LinearRegression()
reg.fit(X, y)

print(f"R^2 : {np.round(reg.score(X, y), 3)}")

print(f"poids = {np.round(reg.coef_[0], 2)} * sexe + {np.round(reg.coef_[1], 2)} * age + bruit")

# les variables prédictives
X2 = df[['sexe', 'age', 'taille']]

# la variable cible, le poids
y2 = df.poids

# entrainons un nouveau modele de regression lineaire
reg2 = LinearRegression()
reg2.fit(X2, y2)



# le score
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score
# The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
# A constant model that always predicts the expected value of y, disregarding the input features, would get a score of 0.0.
print(f"R^2 : {np.round(reg2.score(X2, y2), 3)}")
# et les coefficients
#  le bruit représente l'information qui n'est pas capturée par le modèle linéaire
print(f"poids = {np.round(reg2.coef_[0],  2)} * sexe + {np.round(reg2.coef_[1],  2)} * age +  {np.round(reg2.coef_[2], 2)} * taille + du bruit")

# prédiction
poids = reg.predict(pd.DataFrame([[0, 150]], columns=['sexe', 'age']))
poids2 = reg2.predict(pd.DataFrame([[0, 150, 155]], columns=['sexe', 'age', 'taille']))
print(poids)
print(poids2)


X3 = df[['taille']]
y3 = df.poids

reg3 = LinearRegression()
reg3.fit(X3, y3)

print(f"R^2 : {np.round(reg3.score(X3, y3), 3)}")

print(f"poids = {np.round(reg3.coef_[0], 2)} * taille + bruit")

y_pred = reg.predict(X)
y2_pred = reg2.predict(X2)

print(mean_squared_error(y, y_pred))
print(mean_absolute_error(y, y_pred))
print(mean_absolute_percentage_error(y, y_pred))

print(mean_squared_error(y2, y2_pred))
print(mean_absolute_error(y2, y2_pred))
print(mean_absolute_percentage_error(y2, y2_pred))




# data = X2
#
# scaler = StandardScaler()
# data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
#
# print(data.head())
#
# reg3 = LinearRegression()
# reg3.fit(data, y2)
#
# print(f"R^2 : {np.round(reg3.score(data, y2), 3)}")
#
# print(f"poids = {np.round(reg3.coef_[0],  2)} * sexe + {np.round(reg3.coef_[1],  2)} * age +  {np.round(reg3.coef_[2], 2)} * taille + du bruit")

