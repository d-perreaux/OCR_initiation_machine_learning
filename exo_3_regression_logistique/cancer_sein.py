import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# clf pour classifier
clf = LogisticRegression(random_state=808).fit(X_train, y_train)

print(clf.predict([X[8, :]])) # [0]
print(clf.predict_proba([X[8, :]])) # [[0.6900651 0.3099349]]
print(clf.predict([X[13, :]])) #[1]
print(clf.predict_proba([X[13, :]])) # [[0.12361698 0.87638302]]

# Analyser les performances d'un modèle de classification consiste à tracer l'histogramme des probabilités de prediction
y_hat_proba = clf.predict_proba(X_test)[:, 1]

sns.histplot(y_hat_proba)
plt.savefig('histplot.predict_proba.png')

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Matrice de confusion
# [catégories réelles : negatif[TN, FP], positif [FN, TP]]
print(confusion_matrix(y_test, y_pred))
# [[40  3]
#  [ 1 70]]

# Seuil de séparation des classes

y_pred_03 = [0 if value < 0.3 else 1 for value in y_hat_proba]
y_pred_07 = [0 if value < 0.7 else 1 for value in y_hat_proba]

print(confusion_matrix(y_test, y_pred_03))
# [[38  5]
#  [ 1 70]]
print(confusion_matrix(y_test, y_pred_07))
# [[41  2]
#  [ 1 70]]

# Precision, Recall and ROC_AUC
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(roc_auc_score(y_test, y_hat_proba))

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_hat_proba)
plt.figure()
plt.plot(fpr, tpr)
plt.grid()
plt.title("ROC curve")

plt.savefig("roc_curve.png")
