from ucimlrepo import fetch_ucirepo
import pandas as pd

# Télécharger le dataset
dataset = fetch_ucirepo(id=938)

# Sauvegarder les données
X = dataset.data.features
y = dataset.data.targets
df = pd.concat([X, y], axis=1)
df.to_csv('data/raw/appendicitis.csv', index=False)
print("Données sauvegardées dans data/raw/appendicitis.csv")