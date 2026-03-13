# Loan Approval Prediction App

Application web de prédiction d'approbation de prêt utilisant le Machine Learning.

## Fonctionnalités
- Exploration de données interactive
- Prédiction en temps réel
- Visualisations interactives avec Plotly
- Conforme RGPD et Ethics by Design

## Technologies
- Python 3.10
- Streamlit
- Scikit-learn
- Plotly

## Installation locale

```bash
git clone https://github.com/Yacineismael/-Pr-diction-d-Approbation-de-Pr-t.git
cd loan
pip install -r requirements.txt
streamlit run app.py
```

## Structure du projet

```
loan/
├── app.py                          # Application Streamlit
├── requirements.txt                # Dépendances Python
├── loanapproval.csv                # Dataset brut
├── loancleaned.csv                 # Dataset nettoyé
├── modele_logistic_regression.pkl  # Modèle entraîné
├── loaaan.ipynb                    # Notebook exploration & nettoyage
├── loancleaned.ipynb               # Notebook modélisation
└── .streamlit/
    └── config.toml                 # Configuration Streamlit
```

## Dataset

Dataset public — Loan Prediction Dataset (Kaggle)
614 demandes de prêt, 13 variables, cible : `Loan_Status` (Y/N)

## École

NEXA Digital School — M2 Data & Intelligence Artificielle — 2025/2026
