# Implémentez un modèle de scoring

## Table des matières
- [Introduction](#introduction)
- [Strucure du projet](#structure-du-projet)
- [Déroulement du projet](#déroulement-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Déploiement](#déploiement)
- [Auteur](#auteur)

## Introduction
Ce projet a pour objectif de développer un outil de "scoring crédit" pour l'entreprise **"Prêt à dépenser"**. L'objectif est de prédire la probabilité qu'un client rembourse un crédit et de classer sa demande comme "accordée" ou "refusée".

Le projet a été développé en suivant un flux de travail standard :

- **Exploration et modélisation**: Traitement des données et entraînement d'un modèle de Machine Learning pour la prédiction.
- **Versionnement**: Utilisation de Git et GitHub pour gérer le code et l'historique du projet.
- **Déploiement**: Mise en production du modèle de scoring sous forme d'une application web interactive, développée avec Streamlit.


## Structure du projet

Voici l'arborescence du projet, qui contient les scripts d'entraînement, le modèle sauvegardé et l'application Streamlit :

Mon_projet

    ├──data
    │
    │   └──data_cleaned.csv
    │
    │
    ├──.gitattributes
    │
    ├──columns.joblib 
    │
    ├──dashbord.py
    │
    ├──entrainement.py





fhhfjkflflmlmmji
├── notebook
│   └── data
│       └── data_cleaned.csv  # Données nettoyées utilisées pour l'entraînement
├── .gitattributes          # Fichier de configuration Git
├── columns.joblib          # Fichier pour les noms des colonnes du modèle
├── dashbord.py             # Application web Streamlit pour le scoring
├── entrainement.py         # Script Python pour l'entraînement et la sauvegarde du modèle
├── logo.png                # Logo de l'application
├── model.joblib            # Le modèle de Machine Learning sauvegardé
├── readme.md               # Fichier de documentation du projet
└── requirements.txt        # Dépendances Python nécessaires


/votre_projet
    /static
        my_first_app.png
        style.css
    /templates
        main.html
        index.html
        add_user.html
        list_users.html
    api.py
    README.md
    requirements.txt

## Déroulement du projet :

### Phase 1: Entraînement et sauvegarde du modèle
Le script entrainement.py est le cœur de la phase de modélisation. Il se charge du nettoyage des données, de l'entraînement du modèle de scoring, et de sa sauvegarde dans le fichier model.joblib. Le fichier columns.joblib est également créé pour s'assurer que les données d'entrée de l'application Streamlit correspondent aux caractéristiques du modèle.

### Phase 2: Versionnement avec Git 
Tout le code source et les artefacts importants (modèles, données nettoyées, etc.) sont versionnés sur GitHub afin de conserver un historique complet des modifications du projet.

### Phase 3: Déploiement sur Streamlit Community Cloud
Le fichier dashbord.py est l'application web Streamlit qui interagit avec le modèle sauvegardé pour offrir une interface utilisateur intuitive. Cette application a été déployée sur Streamlit Community Cloud, offrant un accès public et une mise à jour automatique à chaque git push.

## Prérequis
- Python 3.x
- Les librairies Python listées dans le fichier requirements.txt.
- Git et un compte GitHub.

## Installation

### Etape 1: Clonage du dépôt
```bash
git clone  https://github.com/soso395/mon_api
cd mon_api
```
### Etape  2 : Installation des dépendances
```bash
pip install -r requirements.txt
```


## Utilisation
Pour lancer l'application Streamlit en local, exécutez la commande suivante depuis la racine du projet :

```bash
streamlit run dashbord.py
```
Une page s'ouvrira dans votre navigateur web, vous permettant d'utiliser l'outil de scoring en local.

## Déploiement
Ce projet est déployé sur Streamlit Community Cloud et est accessible à l'adresse suivante :
 https://monapi-ls9jhgpyxj8y7btrhy4qhn.streamlit.app/

L'application est mise à jour automatiquement à chaque git push sur la branche principale de votre dépôt GitHub.

## Auteur

Développé par saadjamiri. Pour toute question, veuillez contacter saad1.jamiri@gamil.com.
