# NLP Model Comparison: Sentiment Analysis & Authorship Attribution

Comparative study of Machine Learning and Deep Learning models applied to two text classification tasks:

1. **Sentiment analysis** - binary classification on 27,000 movie reviews
2. **Authorship attribution** - identifying the speaking style of French presidents Chirac (49890 sentences) vs. Mitterrand (7523 sentences), a notably imbalanced dataset (~87/13 split)

Built in a pair project with teammate Yuyu CHEN.

## Results

| Task | Best model | F1 score | Runner-up |
|---|---|---|---|
| Sentiment (movies) | DistilBERT | 0.92 | TF-IDF + SVM (0.90) |
| Authorship (speeches) | CamemBERT-large | 0.71 | TF-IDF + SVM (0.65) |

Transformers outperformed all classical approaches on both tasks, with a much larger gap for authorship attribution than for sentiment analysis - suggesting that stylistic classification benefits more from the rich contextual representations Transformers provide.
The authorship task's class imbalance (~87% Chirac / ~13% Mitterrand) made it meaningfully harder than the balanced sentiment task, and motivated the weighted sampling/loss strategies used in the RNN and Transformer models (see `RNNWrapper` and `TransformerWrapper` below).

## Approach

For each task, we benchmarked the impact of multiple preprocessing strategies, vectorization methods, and model architectures - ranging from linear models (TF-IDF + SVM/Logistic Regression) to deep learning (BiLSTM) to fine-tuned Transformers (DistilBERT, CamemBERT).

## Repo structure

- `Codes/Movies/`: sentiment analysis pipeline (`dist.ipynb`, `main_movies.ipynb`, `utils.py`)
- `Codes/Presidents/`: authorship attribution pipeline (`cam.ipynb`, `main_pres.ipynb`, `preprocessing.py`)
- `Results/`: challenge submission files (CSV)

Not included: final trained models, raw train/test source files, report figures.

<details>
<summary><strong>Notebook & pipeline details (click to expand)</strong></summary>

### `dist.ipynb` (`/Codes/Movies`)
Fine-tuning the DistilBERT model on Google Colab:
- `TransformerWrapper` class
- Loading the 25,000-review movie corpus
- Trainer configuration
- Training and predictions on test set

### `cam.ipynb` (`/Codes/Presidents`)
Fine-tuning the CamemBERT model on Google Colab:
- `TransformerWrapper` class encapsulating model logic
- Extended dataset including `mitterrand.txt` (16,737 additional sentences)
- Trainer configuration (hyperparameters, evaluation strategy)
- Training and predictions on test set

### `main_movies.ipynb` (`/Codes/Movies`)
Main exploration notebook for task 1:
- Data loading and exploratory analysis
- TF-IDF + SVM pipeline with `GridSearchCV`
- Automated experiment runs testing different preprocessing functions and models, using functions from `utils.py`

### `main_pres.ipynb` (`/Codes/Presidents`)
Main exploration notebook for task 2:
- Data loading and exploratory analysis
- TF-IDF + SVM pipeline with `GridSearchCV`
- Automated experiment runs testing different preprocessing functions and models (Logistic Regression, SVM, RNN, Transformer), using functions from `preprocessing.py`



### `utils.py`
Core experimental pipeline for task 1 (movies):
- Four preprocessing levels (`clean_raw` to `clean_keep_negation`)
- `ExperimentResult` structure for systematic tracking of metrics (Accuracy, F1-macro, Log-Loss)
- Dynamic construction of Scikit-Learn pipelines (TF-IDF + SVM/LogReg), Keras architectures (BiLSTM), and Transformers (DistilBERT)
- Stratified cross-validation, automatic confusion matrix generation, global best-model selection (`copy_best_models`)
- `predict_with_best_global` function to load the optimal model and generate the final submission CSV

### `preprocessing.py`
Core experimental pipeline for task 2 (presidents):
- Five preprocessing levels (`clean_1` to `clean_4`) plus `preprocess_pres` for SpaCy-based lemmatization
- Model wrappers:
  - `W2VLogRegWrapper` - TF-IDF-weighted Word2Vec with early stopping (underwhelming results, excluded from the report)
  - `RNNWrapper` - PyTorch Bi-LSTM architecture with `WeightedRandomSampler` to handle class imbalance
  - `TransformerWrapper` - HuggingFace CamemBERT integration with a custom loss function (`WeightedTrainer`)
- Metric computation, stratified cross-validation, formatted confusion matrices
- Save/load handling for `.joblib` files (Scikit-Learn) and weight directories (Deep Learning)

</details>
