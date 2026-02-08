# 📊Bank Marketing – Analyse Exploratoire & Aide à la Décision

## 👥 Équipe
- **Reda BOUCHAREB**
- **Clément KAFWIMBI**

----

## 🎯 Objectif
Analyser les données d’une campagne marketing bancaire afin d’identifier  
les clients les plus susceptibles de souscrire à un **dépôt à terme**,  
et fournir des **recommandations stratégiques basées sur les données**.

---

## 📌 Méthodologie
Le projet suit l’approche **CRISP-DM** :

1. Compréhension du métier et des données  
2. Analyse exploratoire des données  
3. Nettoyage et préparation des données  
4. Analyses statistiques  
5. Modélisation simple (KNN)  
6. Recommandations business  
7. Dashboard interactif (bonus)

---

## 📌 Structure du projet

```bash
Mini_Projet_DM1_analyse_exploratoire_Reda_&_Clement/
│
├── data/                           # Répertoire central des ressources
│   ├── clean/                      # Données préparées pour le modèle
│   │   ├── bank_marketing_clean_VF.csv
│   │   ├── bank_marketing_clean_VF_test.csv
│   │   ├── bank_marketing_encoded.csv
│   │   └── bank_marketing_test_encoded.csv
│   │
│   ├── doc/                        # Rapport et livrables
│   │   └── MiniProjet_DM1_analyse_exploratoire.pdf
│   │
│   ├── models/                     # Artefacts du Machine Learning
│   │   ├── knn_model.pkl
│   │   ├── knn_model_complete.pkl
│   │   └── scaler.pkl
│   │
│   └── raw/                        # Sources de données originales
│       ├── bank-full.csv
│       ├── bank.csv
│       └── bank-names.txt
│
├── notebooks/                      # Analyse et entraînement
│   └── analyse_exploratoire_&_modeling_knn.ipynb
│
├── src/                            # Code source modulaire
│   ├── __init__.py
│   ├── common.py                   # Détection des types de variables
│   └── statistics.py               # Tests statistiques (ANOVA, Chi-2)
│
├── app.py                          # Interface de prédiction Streamlit
├── requirements.txt                # Liste des bibliothèques nécessaires
└── README.md                       # Documentation et mode d'emploi
```
## ▶️ Comment exécuter le projet

### 🔹 Prérequis
- **Python 3.12**
- `pip`
- (Optionnel) Git

---

### 1️⃣ Récupération du projet

#### Option 1 : via Git
```bash
git clone <url_du_repository>
cd Mini_Projet_DM1_analyse_exploratoire_Reda_&_Clement 
```

#### Option 2 : via un fichier ZIP
- Télécharger le projet au format ZIP
- Décompresser l’archive
- Ouvrir un terminal dans le dossier du projet

### 2️⃣ Création de l’environnement virtuel
```
python -m venv env
ou
python3 -m venv env
```
### 3️⃣ Activation de l’environnement
#### macOS / Linux
```
source env/bin/activate
```
#### Windows
```
env\Scripts\activate
```

### 4️⃣ Installation des dépendances
```
pip install -r requirements.txt
```
### 5️⃣ Exécution des analyses (Notebooks)
#### Lancer Jupyter Notebook :
```
jupyter notebook
```
#### Puis ouvrir les notebooks dans l’ordre suivant :
- analyse_expolratoire_&__modeling_knn.ipynb
### 6️⃣ Lancer l’application Streamlit (option bonus)
```
streamlit run app.py
```

## ℹ️ Remarques
- Le projet a été testé avec Python 3.12
- Les dépendances sont listées dans le fichier requirements.txt
- Le dossier data/processed contient les données nettoyées utilisées pour l’analyse