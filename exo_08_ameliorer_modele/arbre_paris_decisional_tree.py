import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

filename = 'https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/paris-arbres-numerical-2023-09-10.csv'
data = pd.read_csv(filename)

X = data[['domanialite', 'arrondissement', 'libelle_francais', 'genre', 'espece', 'circonference_cm', 'hauteur_m']]
y = data.stade_de_developpement.values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=808)

clf = DecisionTreeClassifier(
    max_depth=3,
    random_state=808
)
clf.fit(X_train, y_train)

train_auc = roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovr')
test_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')

from sklearn.metrics import roc_auc_score
train_auc = roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovr')
test_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
print("train",train_auc) # train 0.89
print("test", test_auc) # test 0.89

y_train_hat = clf.predict(X_train)
y_test_hat = clf.predict(X_test)

print(confusion_matrix(y_test, y_test_hat))
print(confusion_matrix(y_train, y_train_hat))
# [[vrais classe 1]
#  [vrais classe 2]
#  [vrais classe 3]
#  [vrais classe 4]]
# Donc la diagonale représente les vrais positifs, et tous les autres des faux positifs et faux négatifs spécifique à chaque classe

print(classification_report(y_train, y_train_hat))
print(classification_report(y_test, y_test_hat))

print(clf.score(X_train, y_train))
# 0.731
print(clf.score(X_test, y_test))
# 0.74

clf = DecisionTreeClassifier(
    max_depth=None,
    random_state=808
)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
# 0.95
print(clf.score(X_test, y_test))
# 0.81

# Donc score(train) excellent et score(test) moyen : overfit
# Car trop de complexité

# Solution est entre max_depth = 3 et max_depth = infini
scores = []
for depth in np.arange(2, 30, 2):
    clf = DecisionTreeClassifier(
        max_depth = depth,
        random_state = 808
    )

    clf.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovr')
    test_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
    scores.append({
        'max_depth': depth,
        'train': train_auc,
        'test': test_auc,
    })

scores = pd.DataFrame(scores)
print(scores)

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(1, 1, 1)
plt.plot(scores.max_depth, scores.train, label = 'train')
plt.plot(scores.max_depth, scores.test, label = 'test')
plt.plot(scores.max_depth, scores.train, '*', color = 'gray')
plt.plot(scores.max_depth, scores.test, 'o', color = 'gray')
ax.grid(True, which = 'both')
ax.set_title('Train and test AUC vs max_depth')
ax.set_xlabel('Max depth')
ax.set_ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.savefig('train_and_test_auc_vs_max_depth.png')
