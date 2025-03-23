import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from scipy import stats


dataset_url = "https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/paris-arbres-clean-2023-09-10.csv"
df = pd.read_csv(dataset_url)

print(df.shape)
print(df.head)
print(df.columns)
print(df.describe())
print(df.info())
print(df.domanialite.value_counts(dropna=False))
print(df.espece.value_counts(dropna=False))
print(df.stade_de_developpement.value_counts(dropna=False))


# 221201 échantillons
# 16 colonnes
# les variables suivantes:
#     localisation: domanialite, arrondissement, complement_adresse, numero, lieu_adresse, geo_point_2d
#     nature des arbres: libelle_francais, genre, espece, variete_oucultivar, remarquable
#     les mensurations: circonference_cm et hauteur_m, stade_de_developpement
#     les ID: idbase, idemplacement

# On se limite aux platanes (42 500 arbres)
df = df[df.libelle_francais == 'Platane'].copy()
print(df.shape)
# 3350 NaN
print(df.stade_de_developpement.value_counts(dropna=False))
# On suppose que jeune (arbre) Adulte est un stade intermédiaire entre jeune (arbre) et adulte
# Pour vérifier notre hypothèse, on peut regarder la taille et la circonférence en fct du stade de developpement
#
cats = ['Jeune (arbre)', 'Jeune (arbre)Adulte', 'Adulte', 'Mature']

for n, cat in zip(range(1, 5), cats):
    df.loc[df.stade_de_developpement == cat, 'stade_num'] = n
df.sort_values(by='stade_num', inplace=True)
df.reset_index(inplace=True, drop=True)

# Visualisation de la répartition sauf les NaN
cond = ~df.stade_de_developpement.isna()
plt.figure()
fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(1, 2, 1)
sns.boxplot(data=df[cond], y="circonference_cm", x="stade_de_developpement")
ax.grid(True, which='both')
ax.set_title('distribution de la circonférence par stade de développement')

ax = fig.add_subplot(1, 2, 2)
sns.boxplot(data=df[cond], y="hauteur_m", x="stade_de_developpement")
ax.grid(True, which='both')
ax.set_title('distribution de la hauteur par stade de développement')

plt.tight_layout()
plt.savefig('distribution_circonference_hauteur_fct_stade_developpement.png')

# A partir de ces visualisations,
# nous pouvons choisir des seuils de hauteur et de
# circonference pour determiner le stade de developpement de l'arbre
# Surtout pour les jaunes et matures car ils sont clairement identifiables

# création conditions pour Mature
cond = ~df.stade_de_developpement.isna() & (df.circonference_cm < 400) & (df.hauteur_m < 40)
print(df[cond].shape)

# Création conditions pour Jeune
cond = (df.stade_de_developpement.isna()) & (df.hauteur_m < 8) & (df.circonference_cm < 50)
print(df[cond].shape)



# Nous pouvons aussi utiliser la hauteur et la
# circonference pour determiner le stade de developpement de l'arbre
# via une régression logistique
# afin de déterminer le stade de développement des NaN

cond = ~df.stade_de_developpement.isna()

X = df[cond][['hauteur_m', 'circonference_cm']]
y = df[cond].stade_num

# Scaling
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
print(f"Score de la classification R^2: ", clf.score(X_test, y_test))

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Combien d'arbres dont le stade_de_developpement est manquant seraient classé?
cond = ~df.stade_de_developpement.isna()
X_missing = scaler.transform(df[~cond][['hauteur_m', 'circonference_cm']])
y_missing = clf.predict(X_missing)
print(y_missing)

print(np.unique(y_missing, return_counts=True)[1])
# soit 2913 Jeune arbres, 24 Jeunes arbres (Adulte), 385 Adulte et 32 Mature

# L'étape d'après, plus complexe, serait d'utiliser une régression logistique pour
# identifier le stade de développement à partir des variables significatives, hauteur,
# circonférence mais aussi par exemple la domanialité et l'espèce.

## Détecter les outliers
# Par visualisation
plt.figure()
fig = plt.figure(figsize=(6, 6))
plt.plot(df.circonference_cm, df.hauteur_m, '.')
plt.grid()
plt.xlabel('circonférence (cm)')
plt.ylabel('hauteur (m)')
plt.title('Platanes')
plt.savefig('hauteur_egale_f(circonference).png')
# On elève les cas vraiment extremes et quasi uniques à partir de la visualisation
df = df[(df.circonference_cm < 1000) & (df.hauteur_m < 100)].copy()

# Par z-score
df['z_circonference'] = stats.zscore(df.circonference_cm)
df['z_hauteur'] = stats.zscore(df.hauteur_m)
plt.figure()
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)
df.z_circonference.hist(bins = 100)
ax.grid(True, which = 'both')
ax.set_title('circonférence - z-score')
ax = fig.add_subplot(1, 2, 2)
df.z_hauteur.hist(bins = 100)
ax.grid(True, which = 'both')
ax.set_title('hauteur - z-score')
plt.savefig('z_score.png')

# 2 ecart type
print(f"2 ecart type - circonference:", df[df.z_circonference > 2].shape[0], "arbres\tcirconference max:", np.max(df[df.z_circonference < 2].circonference_cm) , "cm" )
print(f"2 ecart type - hauteur:", df[df.z_hauteur > 2].shape[0], "arbres\thauteur max:", np.max(df[df.z_hauteur < 2].hauteur_m) , "m" )
print()
# 3 ecart type
print(f"3 ecart type - circonference:", df[df.z_circonference > 3].shape[0], "arbres\tcirconference max:", np.min(df[df.z_circonference > 3].circonference_cm) , "cm" )
print(f"3 ecart type - hauteur:", df[df.z_hauteur > 3].shape[0], "arbres\thauteur max:", np.min(df[df.z_hauteur > 3].hauteur_m) , "m" )

# Méthode de l'IQR
iqr = np.quantile(df.hauteur_m, q=[0.25, 0.75])
limite_basse = iqr[0] - 1.5 * (iqr[1] - iqr[0])
limite_haute = iqr[1] + 1.5*(iqr[1] - iqr[0])

print("limite_haute hauteur:", limite_haute)
print("limite_basse hauteur:", limite_basse)

# Prendre le log de la variable pour atténuer la dispersion


# toujours rajouter 1 a une variable positive quand on prend son log pour eviter le log(0)
df['log_circonference'] = np.log(df.circonference_cm + 1)
df['z_circonference_log'] = stats.zscore(df.log_circonference)

# z-score
print(f"2 ecart type - circonference:", df[df.z_circonference_log > 2].shape[0], "arbres\tcirconference max:", np.min(df[df.z_circonference_log > 2].circonference_cm) , "cm" )
print(f"3 ecart type - circonference:", df[df.z_circonference_log > 3].shape[0], "arbres\tcirconference max:", np.min(df[df.z_circonference_log > 3].circonference_cm)  )

plt.figure()
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)
df.circonference_cm.hist(bins = 100)
ax.grid(True, which = 'both')
ax.set_title('circonférence')
ax = fig.add_subplot(1, 2, 2)
df.log_circonference.hist(bins = 100)
ax.grid(True, which = 'both')
ax.set_title('log circonference')
plt.savefig('log_circonference.png')
# histogramme est plus concentré autour de la valeur moyenne de la variable.
# L'effet des outliers est donc presque entièrement compensé

# Discretiser la variable
# qcut va scinder la variable en intervalles de volume sensiblement égales en fonction de leur fréquence
df['hauteur_qcut'] = pd.qcut(df.hauteur_m, 4, labels=[1,2,3, 4])
data = df.hauteur_qcut.value_counts()

plt.figure()
fig, ax = plt.subplots()
ax.bar(data.index, data.values, label= data.index)
plt.savefig('qcut.png')

# cut va scinder les données en intervalles de meme taille
df['hauteur_cut'] = pd.cut(df.hauteur_m, 3, labels=["petit", "moyen", "grand"])
data = df.hauteur_cut.value_counts()

plt.figure()
fig, ax = plt.subplots()
ax.bar(data.index, data.values, label= data.index)
plt.savefig('cut.png')


## Standardization
scaler = StandardScaler()
df['hauteur_standard'] = scaler.fit_transform(df.hauteur_m.values.reshape(-1, 1))
df['circonference_standard'] = scaler.fit_transform(df.circonference_cm.values.reshape(-1, 1))

## Log
# Toujours ajouter +1 avant de prendre le log pour éviter les log(0)
df['circonference_log'] = np.log(df.circonference_cm + 1)
df['hauteur_log'] = np.log(df.hauteur_m + 1)
