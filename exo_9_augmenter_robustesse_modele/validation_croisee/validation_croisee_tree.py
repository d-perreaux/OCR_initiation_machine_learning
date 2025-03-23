import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


filename = 'https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/paris-arbres-numerical-2023-09-10.csv'
data = pd.read_csv(filename)

X = data[['domanialite', 'arrondissement', 'libelle_francais', 'genre', 'espece', 'circonference_cm', 'hauteur_m']]
y = data.stade_de_developpement.values



# On entraine le modele en faisant ue validation croisée à 5 plis (fold)
## On entraine le modèle sur tout le dataset et non pas sur sous-ensemble car la validation croisée se charge de gérer la séparation
## entraintement / test
## Comme 14 valeurs potentielles du paramètre max_depth et validation croisée à 5 plis
## On entraine 4 * 10 = 70 modèles

parameters = {'max_depth': np.arange(2, 30, 2)}

model = DecisionTreeClassifier(
    random_state=808
)
clf = GridSearchCV(model, parameters, cv = 5, scoring='roc_auc_ovr', verbose=1)
clf.fit(X, y)

print(clf.best_params_)
print(clf.best_score_)
print(clf.best_estimator_)
print(clf.cv_results_)
