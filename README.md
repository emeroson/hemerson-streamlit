# ⚽ Player Efficacy Score — Premier League 2023/2024

> Application interactive de scoring d'efficacité des attaquants de Premier League,  
> construite avec Streamlit et un modèle ElasticNet Regression.



## 📌 Description

Cette application permet de prédire la **note d'efficacité offensive** d'un joueur de Premier League à partir de ses statistiques individuelles. Le score est calibré sur la saison 2023/2024, avec **Erling Haaland** comme référence maximale (2.89).

Le modèle utilisé est une **régression ElasticNet** entraînée sur 52 joueurs offensifs, avec un score R² de **0.83**.



## 🚀 Lancement rapide

```bash
# 1. Cloner le projet
git clone <url-du-repo>
cd <nom-du-repo>

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
streamlit run app.py
```

> ⚠️ Le fichier `model.pkl` doit être présent dans le même répertoire que `app.py`.



## 📁 Structure du projet


├── app.py            # Application Streamlit principale
├── model.pkl         # Modèle ElasticNet + Scaler (joblib bundle)
├── requirements.txt  # Dépendances Python
└── README.md         # Documentation
```



## 🧠 Features du modèle

| Variable | Description |
|---|---|
| `Buts_Par_90` | Nombre de buts marqués par 90 minutes |
| `Passes_Decisives_Par_90` | Nombre de passes décisives par 90 minutes |
| `Contributions_But_Par_90` | Total buts + passes décisives par 90 minutes |
| `Minutes_Par_But` | Minutes jouées entre chaque but |
| `Minutes_Par_Passe_Decisive` | Minutes jouées entre chaque passe décisive |



## 🎯 Interprétation du score

| Score | Niveau | Description |
|---|---|---|
| ≥ 2.00 | 🌟 ÉLITE | Parmi les meilleurs attaquants de PL |
| ≥ 1.50 | ✅ TRÈS BON | Au-dessus de la moyenne, niveau Salah |
| ≥ 1.00 | 📈 DANS LA MOYENNE | Profil correct, marge de progression |
| < 1.00 | ⚠️ EN DESSOUS | En deçà de la moyenne Premier League |

**Référence Top 5 saison 2023/2024 :**

| # | Joueur | Score |
|---|---|---|
| 1 | E. Haaland | 2.89 |
| 2 | C. Palmer | 2.21 |
| 3 | A. Isak | 1.98 |
| 4 | M. Salah | 1.85 |
| 5 | O. Watkins | 1.72 |



## ⚙️ Modèle

- **Algorithme** : ElasticNet Regression
- **Alpha** : 0.1
- **L1 Ratio** : 0.5
- **R²** : 0.83
- **RMSE** : 0.245
- **Dataset** : 52 joueurs offensifs · Premier League 2023/2024
- **Preprocessing** : StandardScaler (inclus dans `model.pkl`)



## 🖥️ Fonctionnalités de l'app

- Saisie des statistiques du joueur via des inputs interactifs
- Prédiction instantanée avec animation de révélation
- Analyse vocale TTS en français (Web Speech API)
- Radar chart comparatif : joueur vs Haaland vs Moyenne PL
- Classement Top 5 de référence avec barres de progression
- Champ nom du joueur personnalisable



## 📦 Dépendances

```
streamlit>=1.32.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.4.0
joblib>=1.3.0
plotly>=5.20.0

## lien de l'app

https://hemerson-app.streamlit.app/

## 👤 Auteur

**ANOH AMON FRANCKLIN HEMERSON**  
Projet Data Science · Master · 2026
