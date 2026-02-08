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
├── data/
│   ├── raw/                     # Données brutes
│   │   ├── bank-additional-full.csv
│   │   └── bank-additional-names.txt
│   │
│   └── processed/               # Données nettoyées
│       └── bank_clean.csv
│
├── notebooks/                   # Analyses exploratoires & statistiques
│   ├── analyse_expolratoire_&__modeling_knn.ipynb
│   
│
├── src/                         # Code Python réutilisable
│   ├── __init__.py
│   └── statistics.py            # Classe StatisticalTests
│
├── app.py                       # Application Streamlit
│
├── README.md                    # Documentation du projet
└── requirements.txt             # Dépendances Python
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