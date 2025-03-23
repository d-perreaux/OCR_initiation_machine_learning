import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


X, y = make_classification(n_classes=2, n_samples=1000, n_features=3, n_informative=2, n_redundant=0, random_state=808)

# Visualisation
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X[:, 0], X[:, 1], c=y)
plt.grid()
plt.savefig('visualisation_dataset.png')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

print(f"Score de la classification R^2: ", clf.score(X_test, y_test))

# Création de jeux de données non linéairement séparables
# Make circle
X_circle, y_circle = make_circles(n_samples=1000, factor=0.3, noise=0.05, random_state=0)

plt.figure()
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X_circle[:, 0], X_circle[:, 1], c=y_circle)
plt.grid()
plt.savefig('make_circles.png')

# Make Moon

X_moon, y_moon = make_moons(n_samples=1000,  noise=0.05, random_state=0)

plt.figure()
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X_moon[:, 0], X_moon[:, 1], c=y_moon)
plt.grid()
plt.savefig('moon.png')

# Régression logistique sur les cercles
X_train, X_test, y_train, y_test = train_test_split(X_circle, y_circle, random_state=0)

clf = LogisticRegression(random_state=0).fit(X_train,y_train)

print(f"Score de la classification R^2 sur les cercles: ", clf.score(X_test, y_test))
plt.figure()
fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 2, 1)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
ax.set_title('original')

#
y_pred = clf.predict(X_test)
ax = fig.add_subplot(1, 2, 2)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
ax.set_title('classification')
plt.savefig('regression_logistic_circles.png')

# Régression logistique sur moons
X_train, X_test, y_train, y_test = train_test_split(X_moon, y_moon, random_state=0)

clf = LogisticRegression(random_state=0).fit(X_train,y_train)

print(f"Score de la classification R^2 sur les moons: ", clf.score(X_test, y_test))
plt.figure()
fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 2, 1)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
ax.set_title('original')

y_pred = clf.predict(X_test)
ax = fig.add_subplot(1, 2, 2)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
ax.set_title('classification')
plt.savefig('regression_logistic_moons.png')
