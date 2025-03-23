import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


filename = 'https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/paris-arbres-numerical-2023-09-10.csv'
data = pd.read_csv(filename)

X = data[['domanialite', 'arrondissement','libelle_francais', 'genre', 'espece','circonference_cm', 'hauteur_m']]
y = data.stade_de_developpement.values

X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=0.8, random_state=808)

clf = RandomForestClassifier(
    n_estimators=100,
    random_state=8
    )

clf.fit(X_train, y_train)

print("test :", np.round(clf.score(X_test, y_test), 3))
print('train:', np.round(clf.score(X_train, y_train), 3))

# Importance relative de chaque feature
print(clf.feature_importances_)

df = pd.DataFrame()
df['feature'] = X.columns
df['importance'] = clf.feature_importances_
df.sort_values(by='importance', ascending=False, inplace=True)

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(1, 1, 1)

sns.barplot(data=df, x='feature', y='importance')
ax.set_title('Feature importance')
ax.set_xlabel('Variable')
ax.set_ylabel('Importance')
ax.grid(True, which='both')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('importance_relative_variables.png')
