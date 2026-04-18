"""
save_model.py
─────────────────────────────────────────────────────────────────────────────
Entraînement ElasticNet + StandardScaler sur Premier_League_fr.csv
Sauvegarde du bundle dans model.pkl

Exécution : python save_model.py
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ─── 1. Chargement des données réelles ───────────────────────────────────────
df = pd.read_csv("Premier_League_fr.csv")
print(f"✅ Données chargées : {df.shape[0]} joueurs, {df.shape[1]} colonnes")

# ─── 2. Features & cible ─────────────────────────────────────────────────────
FEATURES = [
    "Buts_Par_90",
    "Passes_Decisives_Par_90",
    "Contributions_But_Par_90",
    "Minutes_Par_But",
    "Minutes_Par_Passe_Decisive",
]
TARGET = "Note_Efficacite"

X = df[FEATURES]
y = df[TARGET]

print(f"   Note min : {y.min():.3f}  |  max : {y.max():.3f}  |  moyenne : {y.mean():.3f}")

# ─── 3. Split train / test ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── 4. Normalisation ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit uniquement sur train
X_test_scaled  = scaler.transform(X_test)

# ─── 5. Modèle ElasticNet ────────────────────────────────────────────────────
en_model = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    random_state=42,
)
en_model.fit(X_train_scaled, y_train)

# ─── 6. Évaluation ───────────────────────────────────────────────────────────
y_pred = en_model.predict(X_test_scaled)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)
print(f"📊 Performances  —  RMSE : {rmse:.4f}  |  R² : {r2:.4f}")

# ─── 7. Sauvegarde ───────────────────────────────────────────────────────────
joblib.dump({"model": en_model, "scaler": scaler}, "model.pkl")
print("💾 Bundle sauvegardé → model.pkl  (model + scaler)")
