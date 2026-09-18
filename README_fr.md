# Analyse de Sentiment & Attribution Stylistique - Comparaison de Modèles NLP

Étude comparative de modèles de Machine Learning et Deep Learning appliqués à deux tâches de classification de texte :

1. **Analyse de sentiment** : classification binaire sur 27 000 critiques de films
2. **Attribution stylistique** : reconnaissance du style oratoire de Chirac (49 890 phrases) vs Mitterrand (7 523 phrases), un jeu de données fortement déséquilibré (~87/13)

Projet réalisé en binôme avec Yuyu CHEN.

## Résultats

| Tâche | Meilleur modèle | Score F1 | Suivant |
|---|---|---|---|
| Sentiment (films) | DistilBERT | 0.92 | TF-IDF + SVM (0.90) |
| Style oratoire (présidents) | CamemBERT-large | 0.71 | TF-IDF + SVM (0.65) |

Les Transformers surpassent toutes les approches classiques sur les deux tâches, avec un écart bien plus marqué pour l'attribution stylistique que pour l'analyse de sentiment - la reconnaissance du style semble bénéficier davantage des représentations contextuelles riches qu'offrent les Transformers.
Le déséquilibre des classes de la tâche d'attribution stylistique (~87% Chirac / ~13% Mitterrand) a rendu cette tâche nettement plus difficile que l'analyse de sentiment (équilibrée), et justifie les stratégies de pondération (sampling/loss) utilisées dans les modèles RNN et Transformer (voir `RNNWrapper` et `TransformerWrapper` ci-dessous).

## Approche

Pour chaque tâche, nous avons évalué l'influence de plusieurs stratégies de prétraitement, de vectorisation et d'architectures de modèles, allant des approches linéaires (TF-IDF + SVM/Régression logistique) aux réseaux de neurones profonds (BiLSTM), jusqu'aux Transformers fine-tunés (DistilBERT, CamemBERT).

## Structure du répertoire

- `Codes/Movies/` : pipeline d'analyse de sentiment (`dist.ipynb`, `main_movies.ipynb`, `utils.py`)
- `Codes/Presidents/` : pipeline d'attribution stylistique (`cam.ipynb`, `main_pres.ipynb`, `preprocessing.py`)
- `Results/` : soumissions du challenge (CSV)

Non inclus : modèles finaux entraînés, fichiers sources train/test, figures du rapport.

<details>
<summary><strong>Détails des notebooks et pipelines (cliquer pour développer)</strong></summary>

### `cam.ipynb` (`/Codes/Presidents`)
Fine-tuning du modèle CamemBERT sur Google Colab :
- Définition de la classe `TransformerWrapper` pour encapsuler la logique du modèle
- Intégration du jeu de données étendu incluant `mitterrand.txt` (16 737 phrases supplémentaires)
- Configuration du Trainer (hyperparamètres, stratégie d'évaluation)
- Entraînement et génération des prédictions

### `dist.ipynb` (`/Codes/Movies`)
Fine-tuning du modèle DistilBERT sur Google Colab :
- Définition de la classe `TransformerWrapper`
- Chargement du corpus de 25 000 critiques de films
- Configuration du Trainer
- Entraînement et génération des prédictions

### `main_pres.ipynb` (`/Codes/Presidents`)
Notebook principal d'exploration pour la tâche 2 :
- Chargement et analyse exploratoire des données
- Pipeline TF-IDF + SVM avec `GridSearchCV`
- Campagnes d'expériences automatisées testant différentes fonctions de prétraitement et modèles (Régression logistique, SVM, RNN, Transformer), via les fonctions de `preprocessing.py`

### `main_movies.ipynb` (`/Codes/Movies`)
Notebook principal d'exploration pour la tâche 1 :
- Chargement et analyse exploratoire des données
- Pipeline TF-IDF + SVM avec `GridSearchCV`
- Campagnes d'expériences automatisées testant différentes fonctions de prétraitement et modèles, via les fonctions de `utils.py`

### `utils.py`
Cœur du pipeline expérimental pour la tâche 1 (films) :
- Quatre niveaux de prétraitement (`clean_raw` à `clean_keep_negation`)
- Structure `ExperimentResult` pour le suivi systématique des métriques (Accuracy, F1-macro, Log-Loss)
- Construction dynamique de pipelines Scikit-Learn (TF-IDF + SVM/LogReg), d'architectures Keras (BiLSTM) et de Transformers (DistilBERT)
- Validation croisée stratifiée, génération automatique de matrices de confusion, sélection du meilleur modèle global (`copy_best_models`)
- Fonction `predict_with_best_global` pour charger le modèle optimal et générer le CSV de soumission final

### `preprocessing.py`
Cœur du pipeline expérimental pour la tâche 2 (présidents) :
- Cinq niveaux de prétraitement (`clean_1` à `clean_4`) et `preprocess_pres` pour la lemmatisation via SpaCy
- **Wrappers de modèles :**
  - `W2VLogRegWrapper` - Word2Vec pondéré par TF-IDF avec early stopping (performances peu convaincantes, non inclus dans le rapport)
  - `RNNWrapper` - architecture Bi-LSTM sous PyTorch avec `WeightedRandomSampler` pour gérer le déséquilibre des classes
  - `TransformerWrapper` - intégration de CamemBERT via HuggingFace, avec fonction de perte personnalisée (`WeightedTrainer`)
- Calcul de métriques, validation croisée stratifiée, matrices de confusion formatées
- Sauvegarde/chargement des fichiers `.joblib` (Scikit-Learn) et des répertoires de poids (Deep Learning)

</details>
