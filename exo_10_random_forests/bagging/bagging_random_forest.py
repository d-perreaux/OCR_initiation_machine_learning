import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_hastie_10_2

X, y = make_hastie_10_2(n_samples=12000, random_state=808)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=8)

print(X.shape)

# Augmenter le nombre d'arbres (réduire le biais)

tree_counts = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200]
accuracy = []
for n_estimator in tree_counts:
    clf = RandomForestClassifier(
        n_estimators=n_estimator,
        max_depth=2,
        max_features=3,
        random_state=8
    )
    clf.fit(X_train, y_train)
    accuracy.append(clf.score(X_test, y_test))

    print(
        f"{n_estimator} trees \t accuracy test: {np.round(clf.score(X_test, y_test), 3)} \t accuracy train {np.round(clf.score(X_train, y_train), 3)}", )

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 1, 1)
plt.plot(tree_counts, accuracy)
plt.plot(tree_counts, accuracy,'*')
ax.grid(True, which = 'both')
ax.set_title('Accuracy on test vs n_estimators')
ax.set_xlabel('n_estimators')
ax.set_ylabel('Accuracy')
ax.set_ylim(0.9 * np.min(accuracy), 1.1 * np.max(accuracy))
# plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('accuracy_on_tests_vs_nèestimators')

# Réduction de la varianec
# Chaque modèle d'arbre n'a plus de contrainte de profondeur

tree_counts = [1,2,3,4,5,10,15,20,25,30,40,50, 60, 70, 80, 90, 100, 110, 120, 150]

accuracy  = []

for n_estimator in tree_counts:
    clf = RandomForestClassifier(
        n_estimators = n_estimator,
        max_depth = None,
        max_features = None,
        random_state = 8
        )

    clf.fit(X_train, y_train)
    accuracy.append({
        'n': n_estimator,
        'test': clf.score(X_test, y_test),
        'train': clf.score(X_train, y_train),
    })

    print(f"{n_estimator} trees \t accuracy test: {np.round(clf.score(X_test, y_test), 3)} \t accuracy train {np.round(clf.score(X_train, y_train), 3)}", )


accuracy = pd.DataFrame(accuracy)
accuracy['delta'] = np.abs(accuracy.train - accuracy.test)

fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(1, 2, 1)
plt.plot(accuracy.n, accuracy.train, label = 'score train')
plt.plot(accuracy.n, accuracy.train,'*')

plt.plot(accuracy.n, accuracy.test, label = 'score test')
plt.plot(accuracy.n, accuracy.test,'*')

# plt.plot(accuracy.n, accuracy.delta, label = 'delta')
# plt.plot(accuracy.n, accuracy.delta,'*')

ax.grid(True, which = 'both')
ax.set_title('Accuracy')
ax.set_xlabel('n_estimators')
ax.set_ylabel('Accuracy')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.set_ylim(0.9 * np.min(accuracy), 1.1 * np.max(accuracy))

ax.legend()
# --
ax = fig.add_subplot(1, 2, 2)
plt.plot(accuracy.n, accuracy.delta, label = 'delta')
plt.plot(accuracy.n, accuracy.delta,'*')

ax.grid(True, which = 'both')
ax.set_title('Différence score(test) - score(train) ')
ax.set_xlabel('n_estimators')
ax.set_ylabel('Différence score(test) - score(train)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.set_ylim(0.9 * np.min(accuracy), 1.1 * np.max(accuracy))

ax.legend()
plt.tight_layout()
fig.savefig('accuracy_and_delta_sore_vs_n_estimator(no_max_depth).png')

# Estimer l'importance de chaque variable

