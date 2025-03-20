import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from category_encoders.ordinal import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

dataset_url = "https://raw.githubusercontent.com/OpenClassrooms-Student-Center/8063076-Initiez-vous-au-Machine-Learning/master/data/paris-arbres-clean-2023-09-10.csv"
data = pd.read_csv(dataset_url)
df = data[data.libelle_francais == 'Platane'].copy()
# Remarquable est binaire
# remarquable
# NON    41316
# OUI       36
print(df.remarquable.value_counts())

df.loc[df.remarquable == 'NON', 'remarquable'] = 0
df.loc[df.remarquable == 'OUI', 'remarquable'] = 1

print(df.remarquable.value_counts())

# Encodage ordonné
# print(df.stade_de_developpement.value_counts())
# categories = ['Jeune (arbre)', 'Jeune (arbre)Adulte','Adulte', 'Mature']
# for n, categorie in zip(range(1, len(categories)+1), categories):
#     df.loc[df.stade_de_developpement == categorie, 'stade_de_developpement'] = n
#
# print(df.stade_de_developpement.value_counts())

# Encodage ordonnée en utilisant un mapping de category_encoders
df = data[data.libelle_francais == 'Platane'].copy()


mapping = [{'col': 'stade_de_developpement', 'mapping': {np.nan: 0, 'Jeune (arbre)': 1, 'Jeune (arbre)Adulte': 2, 'Adulte': 3, 'Mature': 4}}]

encoder = OrdinalEncoder(mapping=mapping)
stade = encoder.fit_transform(df.stade_de_developpement)
print(stade.value_counts(dropna=False))

# Encodage one-hot
df = df[~df.domanialite.isna()].copy()
df.reset_index(drop=True, inplace=True)

enc = OneHotEncoder()
labels = enc.fit_transform(df.domanialite.values.reshape(-1, 1)).toarray()
print(labels.shape)
new_columns = [f"is_{col.lower()}" for col in df.domanialite.unique()]
print(new_columns)
df = pd.concat([df, pd.DataFrame(columns=new_columns[:-1], data=labels[:, :-1])], axis=1)

for col in new_columns[:-1]:
    df[col] = df[col].astype(int)

df[['domanialite'] + new_columns[:-1]].sample(5, random_state=88)
