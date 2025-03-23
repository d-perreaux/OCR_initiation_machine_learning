# dataset : 200 echantillons sur le budget alloué aux publicités télévisées, radio et journaux
# ainsi que les ventes résultantes

# variable cible : ventes ( variable continue donc bien regression linéaire)
# prédicteurs : tv, radio, journaux

# Tenter de prédire le volume de vente en fonction du budget publicitaire en tv, radio et journaux

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm


df = pd.read_csv('data/advertising.csv')
print(df.head())
print(df.describe())
print(df.info())

# Création des regressions croisées
sns.pairplot(df, kind='reg')
plt.savefig('pairplot.png')

# Création de 3 graphs de régression
fig, ax = plt.subplots(1,3, figsize = (15,5))

plt.subplot(1,3,1)
sns.regplot(x = df[['tv']],y  =  df.ventes)
plt.ylabel('ventes')
plt.xlabel('TV')
plt.title('ventes = a * tv  + b')
plt.grid()
sns.despine()

plt.subplot(1,3,2)
sns.regplot(x = df[['radio']],y  =  df.ventes)
plt.ylabel('ventes')
plt.xlabel('radio')
plt.title('ventes = a * radio + b')
plt.grid()
sns.despine()

plt.subplot(1,3,3)
res = sns.regplot(x = df[['journaux']],y  =  df.ventes)
plt.ylabel('ventes')
plt.xlabel('journaux')
plt.title('ventes = a * journaux + b')
plt.grid()
sns.despine()

plt.tight_layout()
plt.savefig('regplot.png')

# Création d'un tableau avec les coefficients de corrélations croisés entre les variables
print(df.corr())

reg = LinearRegression()

# Séparation du dataset en Train / Test via train_test_split

X = df[['tv', 'radio', 'journaux']]
y = df.ventes

# random_state permet de controler la reproductibilité des résultats
# Les résultats sont alors reproductibles
# Et les modèles comparables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrainer le modèle
reg.fit(X_train, y_train)

# Prédiction de X_test pour tester la performance sur le sous-ensemble de test
y_pred_test = reg.predict(X_test)

# RMSE et MAPE sont les métriques utilisées pour juger les modèles
# RMSE et MAPE : plus le nombre est petit, mieux s'est
# MAPE : entre 0 et 1
# RMSE : pas de contraintes
print(f"RMSE : {mean_squared_error(y_test, y_pred_test)}")  # 3.17
print(f"MAPE : {mean_absolute_percentage_error(y_test, y_pred_test)}") # 0.15


# AMELIORATION 1 : AJOUTER UN TERME QUADRATIQUE
# Besoin de normaliser les variables entre 0 et 1 avec MinMaxScaler
# car l'amplitude de tv2 va être nettement supérieure à celle des autres variables

# Création tv2
df['tv2'] = df.tv**2

# Instanciation
scaler = MinMaxScaler()
# Calcule des min / max des variables
scaler.fit(df)
#Transformation des données
# Commenté pour pas interférer avec boucle en fin de script
# data_array = scaler.transform(df)
# On peut regrouper ces étapes avec la méthode fit_transform(df)
# data_array = fit_transform(df)

# transformation de l'array en dataframe
# df = pd.DataFrame(data_array, columns=['tv', 'radio', 'journaux', 'ventes', 'tv2'])
#
# print(df.describe().loc[['min', 'max']])
#
#
# X = df[['tv', 'radio', 'journaux', 'tv2']]
# y = df.ventes
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# reg.fit(X_train, y_train)
# y_hat_test = reg.predict(X_test)
#
# print(f"Coefficients : {reg.coef_}")
# print(f"RMSE : {mean_squared_error(y_test, y_hat_test)}")
# print(f"MAPE : {mean_absolute_percentage_error(y_test, y_hat_test)}")
#
df['tv_radio'] = df.radio * df .tv

regressions = {
    'simple: y ~ tv + radio + journaux': ['tv', 'radio', 'journaux'],
    'quadratique: y ~ tv + radio + journaux + tv2': ['tv', 'radio', 'journaux', 'tv2'],
    'terme croisée: y ~ tv + radio + journaux + tv*radio': ['tv', 'radio', 'journaux', 'tv_radio']
}

for title, variables in regressions.items():
    scaler = MinMaxScaler()
    data_array = scaler.fit_transform(df[variables])
    # df_scaled = pd.DataFrame(data_array, columns=variables)
    # X = df_scaled[variables]
    X_train, X_test, y_train, y_test = train_test_split(data_array, y, test_size=0.2, random_state=42)

    reg.fit(X_train, y_train)
    y_pred_test = reg.predict(X_test)

    print(f"\n-- Régression {title}")
    print(f"\tRMSE : {mean_squared_error(y_test, y_pred_test)}")
    print(f"\tMAPE : {mean_absolute_percentage_error(y_test, y_pred_test)}")

# -- Régression simple: y ~ tv + radio + journaux
# 	RMSE : 3.1740973539761077
# 	MAPE : 0.15198846602831229
#
# -- Régression quadratique: y ~ tv + radio + journaux + tv2
# 	RMSE : 2.378317089554026
# 	MAPE : 0.13079146046727744
#
# -- Régression terme croisée: y ~ tv + radio + journaux + tv*radio
# 	RMSE : 0.8144305830812211
# 	MAPE : 0.07328855467044483

# RANDOM_STATE et TEST_SIZE

train_test_ratio = [0.8, 0.6, 0.4, 0.2]
random_seeds = [n for n in range(0, 200, 1)]

X = df[['tv', 'radio', 'journaux']]
y = df.ventes

scores = []

for ratio in tqdm(train_test_ratio):
    for seed in random_seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=seed)

        reg.fit(X_train, y_train)
        # Prediction sur le test set
        y_pred_test = reg.predict(X_test)

        scores.append({
            'ratio': ratio,
            'seed': seed,
            'rmse': mean_squared_error(y_test, y_pred_test)
        })
scores = pd.DataFrame(scores)


fig = plt.figure(figsize=(10, 6))
sns.boxplot(x='ratio', y='rmse', hue='ratio', data=scores)
sns.despine()
plt.title("Variation du score en fonction de random_state et du % de données de test")
plt.grid()
plt.savefig('random_state_test_size.png')


# La variabalité du score est plus importante quand le nombre d'échantillons d'entrainement est faible

