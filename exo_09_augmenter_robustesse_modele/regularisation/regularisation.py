import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

random_state = 3

X, y = make_regression(n_samples=30, n_features=1, noise=40, random_state=random_state)
input = np.linspace(np.min(X), np.max(X), 100)

## Simple régression linéaire
model = Ridge(alpha=0)
model.fit(X, y)
y_pred = model.predict(X)

fig = plt.figure( figsize=(9, 6))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X[:, 0], y, label=f"Données source")
model = Ridge(alpha=0)
model.fit(X, y)

y_pred = model.predict(input.reshape(-1, 1))
ax.plot(input, y_pred, label=f"Simple régression linéaire")

ax.set_axis_off()
ax.set_title("Simple régression linéaire")
plt.tight_layout()
plt.savefig("simple_regression_sans_regulateur")
# Ici pas d'overfit, mais une sous-performance


## Overfit via une régression polynomiale de degré 12
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X[:,0], y, label=f"Données source")
model = Ridge(alpha=0)
model.fit(X, y)
y_pred = model.predict(input.reshape(-1, 1))
ax.plot(input, y_pred, label=f"Simple régression linéaire")

# Création de la matrice des prédicteurs
pol = PolynomialFeatures(12, include_bias=False)
XX = pol.fit_transform(X)
px = pol.transform(input.reshape(-1, 1))

# Sans régularisation
model = Ridge(alpha=0)
model.fit(XX, y)

y_pred = model.predict(px)
ax.plot(input, y_pred, label=f"Régression polynômiale degré 12, alpha = 0")

ax.set_axis_off()
ax.legend()
ax.set_title("Le modèle polynomial de degré 12 overfit")
plt.tight_layout()
plt.savefig("regression_polynomiale_sans_regulateur")
# Le modèle overfit les données

## La régularisation
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X[:, 0], y, label=f"Données source")
model = Ridge(alpha=0)
model.fit(X, y)

y_pred = model.predict(input.reshape(-1, 1))
ax.plot(input, y_pred, label=f"Simple régression linéaire")

pol = PolynomialFeatures(12, include_bias=False)
XX = pol.fit_transform(X)
px = pol.transform(input.reshape(-1, 1))

# sans regularisation
model = Ridge(alpha=0)
model.fit(XX, y)

y_pred = model.predict(px)
ax.plot(input, y_pred, label=f"Régression polynômiale degré 12, alpha = 0")

for alpha in [0.0001, 0.001, 0.01, 0.1]:
    model = Ridge(alpha=alpha)
    model.fit(XX, y)
    y_pred = model.predict(px)
    ax.plot(input, y_pred, '--', label=f"alpha {alpha}")

ax.set_axis_off()
ax.legend()
ax.set_title("La regularisation atténue l'overfit")
plt.tight_layout()
plt.savefig("alpha_croissant_attenuation_overfit_via_regulariastion.png")
