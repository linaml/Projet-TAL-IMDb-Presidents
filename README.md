# PROJET TAL 

Ce répertoire contient le code d'une étude comparative de modèles d'apprentissage appliqués à deux tâches distinctes de classification. La première tâche porte sur l’analyse d’un corpus de 27 000 critiques de films classées de façon binaire : sentiment positif ou négatif. La seconde tâche de classification consiste à construire un modèle capable de reconnaître le style linguistique de Chirac et Mitterrand afin d’estimer la probabilité que des phrases aient été dites par l’un ou par l’autre. Pour chaque tâche, nous avons évalué l’influence de plusieurs stratégies de prétraitement, de vectorisation et d’architectures de modèles, allant des approches linéaires aux réseaux de neurones profonds. Pour les deux tâches, le meilleure score en test est obtenu à l’aide des Transformers : le score F1 obtenu est de 0.92 pour la tâche de classification des critiques de films avec le modèle DistilBERT, et de 0.71 pour la reconnaissance de style oratoire, avec Camembert-large. Le couplage de la vectorisation TF-IDF et du SVM linéaire arrive juste derrière pour les films, avec un score F1 de 90.224, et 0.653 pour la seconde tâche.

Ce répertoire comprend :
- le dossier Results: les soumissions faites sur la plateforme du challenge au format CSV ;
- le dossier Codes : les fichiers de code principaux ; 
- ce fichier README.

Ce répertoire ne comprend pas : 
- les modèles finaux ;
- les fichiers sources fournis de train et test pour les deux tâches ;
- les images (matrices et graphiques qui figurent dans le rapport).

### Description des Notebooks

`cam.ipynb` (dossier /Codes/Presidents) :  
Ce fichier est utilisé pour le fine-tuning du modèle CamemBERT avec Google Colab. Il comprend :  
- Importation des bibliothèques ;  
- Définition de la classe `TransformerWrapper` pour encapsuler la logique du modèle ;  
- Intégration du jeu de données étendu incluant le fichier `mitterrand.txt` qui contient 16 737 phrases supplémentaires ;  
- Configuration du Trainer (hyperparamètres et stratégie d'évaluation) ;  
- Phase d'entraînement et génération des prédictions.  

`dist.ipynb` (dossier /Codes/Movies) :  
Ce fichier est utilisé pour le fine-tuning du modèle DistilBERT avec Google Colab. Il comprend :  
- Importation des bibliothèques ;  
- Définition de la classe `TransformerWrapper` pour encapsuler la logique du modèle ;  
- Chargement du corpus de 25 000 critiques de films donné en TME 4 ;  
- Configuration du Trainer (hyperparamètres et stratégie d'évaluation) ;  
- Phase d'entraînement et génération des prédictions.  

`main_pres.ipynb` (dossier /Codes/Presidents) :  
Ce notebook est le notebook principal pour l'exploration et le test des modèles linéaires et RNN de la tâche 2. Il comprend :
- Chargement et courte analyse des données ;  
- Mise en œuvre du pipeline tf-idf + svm avec GridSearchCV ;  
- Exécution de campagnes d'expériences automatisées testant itérativement différentes fonctions de prétraitement et modèles (Régression logistique, SVM, RNN, Transformer), en utilisant des fonctions définies dans `preprocessing.py` que nous décrivons dans la section suivante.  

`main_movies.ipynb` (dossier /Codes/Movies) :  
Ce notebook est le notebook principal pour l'exploration et le test des modèles linéaires et RNN de la Tâche 1. Il comprend :
- Chargement et courte analyse des données ;  
- Mise en œuvre du pipeline tf-idf + svm avec GridSearchCV ;  
- Exécution de campagnes d'expériences automatisées testant itérativement différentes fonctions de prétraitement et modèles (Régression logistique, SVM, RNN, Transformer), en utilisant des fonctions définies dans `utils.py` que nous décrivons dans la section suivante.  

### Description des fichiers `utils.py` et `preprocessing.py`

`utils.py` :  
L'ensemble des fonctions définies dans ce fichier constitue le cœur du pipeline expérimental construit pour la tâche 1 (Films). Il comprend :
- Fonctions de nettoyage : Définition de quatre niveaux de prétraitement (de clean_raw à clean_keep_negation) ;
- Gestionnaire d'expériences : Implémentation de la structure `ExperimentResult` pour le suivi systématique des métriques (Accuracy, F1-macro, Log-Loss) et des chemins de sauvegarde ;
- Fonctions de construction dynamique de pipelines *Scikit-Learn* (Tfidf + SVM/LogReg), d'architectures Keras (BiLSTM) et de Transformers (DistilBERT) ;
- Outils de validation croisée stratifiée (`StratifiedKFold`), génération automatique de matrices de confusion et algorithme de sélection du meilleur modèle global (`copy_best_models`) ;
- Fonction predict_with_best_global permettant de charger le modèle optimal et de générer le fichier de soumission CSV final.  

`preprocessing.py` :  
L'ensemble des fonctions définies dans ce fichier constitue le cœur du pipeline expérimental construit pour la tâche 2 (Présidents). Il regroupe les utilitaires de traitement et les architectures de modèles. Il comprend :
- Fonctions de nettoyage : Définition de cinq niveau de prétraitement allant de clean_1 à clean_4 et une fonction preprocess_pres qui procède à la lemmatisation via SpaCy  ;
- Wrappers de modèles :
  - `W2VLogRegWrapper` : Implémentation d'un modèle Word2Vec pondéré par TF-IDF avec entraînement itératif et early stopping (performances pas convaincantes donc non inclus dans le rapport) ;
  - `RNNWrapper` : Architecture Bi-LSTM sous PyTorch incluant un `WeightedRandomSampler` pour gérer le déséquilibre des classes ;
  - `TransformerWrapper` : Intégration de CamemBERT via la bibliothèque HuggingFace, avec une personnalisation de la fonction de perte (`WeightedTrainer`) ;
- Fonctions de calcul de métriques (Accuracy, F1, Log-Loss), validation croisée stratifiée et génération de matrices de confusion formatées ;
- Système de sauvegarde et de chargement gérant les fichiers `.joblib` pour Scikit-Learn et les répertoires de poids pour les modèles de Deep Learning.
