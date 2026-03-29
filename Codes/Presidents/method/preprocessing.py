import os
import re
import json
import shutil
import random
import warnings
import numpy as np
import pandas as pd
import unicodedata

import nltk
from nltk.corpus import stopwords

from collections import Counter
from joblib import dump, load

from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from gensim.models import Word2Vec

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

from datasets import Dataset as HFDataset

import spacy 
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp_fr = spacy.load('fr_core_news_sm', disable=['parser', 'ner'])

warnings.filterwarnings("ignore")

nltk.download("stopwords")
STOPWORD = set(stopwords.words("french"))

RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# =========================================================
# IO
# =========================================================

def load_pres(fname, for_train=True):
    """
    Lecture fichier
    Retour (labels, texts) si for_train=True sinon texts
    """
    texts, labels = [], []
    if for_train:
        pattern = re.compile(r"^<[0-9]+:[0-9]+:([CM])>(.*)$")
    else:
        pattern = re.compile(r"^<[0-9]+:[0-9]+>(.*)$")

    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) < 3:
                continue

            match = pattern.match(line)
            if not match:
                continue

            if for_train:
                label, text = match.group(1), match.group(2).strip()
                labels.append(1 if label == "M" else 0)  # 1=Mitterrand, 0=Chirac
            else:
                text = match.group(1).strip()

            texts.append(text)

    return (np.array(labels), texts) if for_train else texts


# =========================================================
# Nettoyage
# =========================================================

def clean_1(txt):
    """
    Nettoyage simple:
    - minuscules
    - suppression balises
    - conservation lettres accentuées et apostrophes / tirets
    - normalisation espaces
    """
    txt = txt.lower()
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s'-]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def preprocess_pres(text):
    # 1. Normalisation
    text = text.lower().strip()
    text = unicodedata.normalize('NFC', text)
    text = text.replace("'", " ").replace("’", " ")
    text = re.sub(r'[^a-zàâçéèêëîïôûù\s]', ' ', text)
    text = re.sub(r"\s+", " ", text).strip()
    def lemmatize_text(text):
        doc = nlp_fr(text)
        return " ".join([token.lemma_ for token in doc])
    lemmatize_text(text)
    
    return text


def clean_2(txt):
    """
    Nettoyage plus agressif:
    - suppression balises
    - suppression ponctuation/chiffres
    - suppression mots très courts
    - suppression stopwords
    """
    txt = txt.lower()
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"['-]", " ", txt)
    txt = re.sub(r"\d+", " ", txt)
    txt = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", " ", txt)
    tok = txt.split()
    tok = [t for t in tok if len(t) > 2 and t not in STOPWORD]
    return " ".join(tok)


def clean_3(txt):
    """
    Variante intermédiaire:
    - conserve structure lexicale
    - enlève chiffres
    - normalise espaces
    """
    txt = txt.lower()
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"\d+", " ", txt)
    txt = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s']", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def clean_4(txt):
    return txt.strip()


# =========================================================
# Métriques
# =========================================================

def compute_metrics(y_true, pred, proba):
    proba = np.asarray(proba)
    proba = np.clip(proba, 1e-8, 1 - 1e-8)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "log_loss": log_loss(y_true, proba),
        "confusion_matrix": confusion_matrix(y_true, pred)
    }


def confusion_matrix_df(cm):
    return pd.DataFrame(
        cm,
        index=["Vrai Chirac", "Vrai Mitterrand"],
        columns=["Prédit Chirac", "Prédit Mitterrand"]
    )


def print_metrics(title, metrics):
    print("=" * 50)
    print(title)
    print("=" * 50)
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"F1-score  : {metrics['f1']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"Log-loss  : {metrics['log_loss']:.4f}")
    print("Confusion matrix :")
    print(confusion_matrix_df(metrics["confusion_matrix"]))


def run_cross_validation(model, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "neg_log_loss": "neg_log_loss"
    }

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    summary = {}
    for metric, values in scores.items():
        if metric.startswith("test_"):
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }
    return summary


def print_cv_results(title, cv_results):
    print("#" * 50)
    print(f"CROSS-VALIDATION : {title}")
    print("#" * 50)
    for metric, stat in cv_results.items():
        print(f"{metric:<18} : {stat['mean']:.4f} +/- {stat['std']:.4f}")


def evaluate_model(model, X_valid, y_valid):
    pred = model.predict(X_valid)
    proba_valid = model.predict_proba(X_valid)
    if proba_valid.ndim == 2:
        proba_valid = proba_valid[:, 1]
    return compute_metrics(y_valid, pred, proba_valid)


# =========================================================
# Modèles classiques
# =========================================================

def tfidf_logreg():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=False,
            analyzer="word",
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.95,
            max_features=80000,
            sublinear_tf=True,
            strip_accents=None
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            C=2.0,
            max_iter=4000,
            solver="liblinear",
            random_state=RANDOM_STATE
        ))
    ])


def tfidf_logreg_char():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=False,
            analyzer="char_wb",
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.98,
            max_features=10000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            C=2.0,
            max_iter=5000,
            solver="saga",
            random_state=RANDOM_STATE
        ))
    ])


def tfidf_svm():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=False,
            analyzer="char_wb",
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.98,
            max_features=10000,
            sublinear_tf=True
        )),
        ("clf", CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                C=1.5,
                random_state=RANDOM_STATE
            ),
            method="sigmoid",
            cv=5
        ))
    ])


# =========================================================
# Word2Vec + LogisticRegression amélioré
# =========================================================

def tokenize_simple(text):
    # enlever accents
    text=unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    # garder seulement lettres + chiffres
    text = re.sub(r"[^a-z0-9\s]", " ", text).strip()
    return text.split()

class W2VLogRegWrapper:
    def __init__(
        self,
        vector_size=200,
        window=5,
        min_count=2,
        sg=1,
        epochs=50,
        clf_c=2.0,
        patience=4,
        monitor="f1",          # "f1" ou "log_loss"
        min_delta=1e-4
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.clf_c = clf_c
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta

        self.w2v = None
        self.tfidf = None
        self.idf_map = None
        self.clf = None

    def _build_clf(self):
        return LogisticRegression(
            class_weight="balanced",
            C=self.clf_c,
            max_iter=4000,
            solver="liblinear",
            random_state=RANDOM_STATE
        )

    def fit(self, texts, y, texts_valid=None, y_valid=None):
        sentences = [tokenize_simple(t) for t in texts]

        # TF-IDF appris uniquement sur train
        self.tfidf = SklearnTfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            min_df=2
        )
        self.tfidf.fit(texts)
        self.idf_map = dict(zip(self.tfidf.get_feature_names_out(), self.tfidf.idf_))

        # Initialisation W2V sans entraînement complet
        self.w2v = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            sg=self.sg,
            seed=RANDOM_STATE
        )
        self.w2v.build_vocab(sentences)

        best_score = -np.inf if self.monitor == "f1" else np.inf
        best_wv = None
        best_clf = None
        patience_left = self.patience

        for epoch in range(self.epochs):
            # entraîne 1 epoch à la fois
            self.w2v.train(
                sentences,
                total_examples=len(sentences),
                epochs=1
            )

            # réentraîne la régression sur les embeddings courants
            X_train_vec = np.vstack([self.text_to_vec(t) for t in texts])
            clf = self._build_clf()
            clf.fit(X_train_vec, y)
            self.clf = clf

            msg = f"[W2V] epoch {epoch+1}/{self.epochs}"

            if texts_valid is not None and y_valid is not None:
                X_valid_vec = np.vstack([self.text_to_vec(t) for t in texts_valid])
                y_pred = clf.predict(X_valid_vec)
                y_prob = clf.predict_proba(X_valid_vec)[:, 1]

                val_f1 = f1_score(y_valid, y_pred)
                val_ll = log_loss(y_valid, np.clip(y_prob, 1e-8, 1 - 1e-8))

                msg += f" - val_f1={val_f1:.4f} - val_log_loss={val_ll:.4f}"

                if self.monitor == "f1":
                    improved = val_f1 > best_score + self.min_delta
                    current_score = val_f1
                else:
                    improved = val_ll < best_score - self.min_delta
                    current_score = val_ll

                if improved:
                    best_score = current_score
                    patience_left = self.patience
                    best_wv = {w: self.w2v.wv[w].copy() for w in self.w2v.wv.index_to_key}
                    best_clf = clf
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(msg)
                        print("[W2V] early stopping")
                        break

            print(msg)

        # restauration du meilleur état
        if best_clf is not None and best_wv is not None:
            for w in best_wv:
                self.w2v.wv[w] = best_wv[w]
            self.clf = best_clf
        else:
            # cas sans validation
            X_train_vec = np.vstack([self.text_to_vec(t) for t in texts])
            self.clf = self._build_clf()
            self.clf.fit(X_train_vec, y)

        return self

    def text_to_vec(self, text):
        tokens = tokenize_simple(text)
        weighted_vectors = []

        for w in tokens:
            if w in self.w2v.wv:
                weight = self.idf_map.get(w, 1.0) if self.idf_map is not None else 1.0
                weighted_vectors.append(self.w2v.wv[w] * weight)

        if not weighted_vectors:
            return np.zeros(self.vector_size, dtype=np.float32)

        vec = np.mean(weighted_vectors, axis=0)
        return vec.astype(np.float32)

    def predict(self, texts):
        X = np.vstack([self.text_to_vec(t) for t in texts])
        return self.clf.predict(X)

    def predict_proba(self, texts):
        X = np.vstack([self.text_to_vec(t) for t in texts])
        return self.clf.predict_proba(X)


def w2v_logreg():
    return W2VLogRegWrapper(
        epochs=50,
        patience=4,
        monitor="f1",
        min_delta=1e-4,
        clf_c=2.0
    )


# =========================================================
# RNN / LSTM amélioré
# =========================================================

def vocabulaires(texts, min_freq=2, max_vocab=30000):
    counter = Counter()
    for text in texts:
        counter.update(tokenize_simple(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab:
            break
        vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_len):
    tokens = tokenize_simple(text)[:max_len]

    # évite longueur 0 pour pack_padded_sequence
    if len(tokens) == 0:
        return [vocab["<UNK>"]] + [vocab["<PAD>"]] * (max_len - 1), 1

    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    length = len(ids)

    if length < max_len:
        ids += [vocab["<PAD>"]] * (max_len - length)

    return ids, length


class TextDatasetRNN(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.X = []
        self.lengths = []
        self.y = labels

        for t in texts:
            ids, length = encode_text(t, vocab, max_len)
            self.X.append(ids)
            self.lengths.append(length)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.X[idx], dtype=torch.long),
            "lengths": torch.tensor(self.lengths[idx], dtype=torch.long),
            "labels": torch.tensor(self.y[idx], dtype=torch.float)
        }


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=64, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids, lengths):
        x = self.embedding(input_ids)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)

        h_forward = hidden[-2]
        h_backward = hidden[-1]
        h = torch.cat((h_forward, h_backward), dim=1)

        h = self.dropout(h)
        logits = self.fc(h).squeeze(1)
        return logits


def clone_state_dict(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


class RNNWrapper:
    def __init__(
        self,
        max_len=48,
        emb_dim=128,
        hidden_dim=64,
        batch_size=64,
        lr=1e-4,
        epochs=50,
        patience=4,
        monitor="f1",      # "f1" ou "loss"
        min_delta=1e-4
    ):
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta

        self.vocab = None
        self.model = None
        self.best_threshold = 0.5

    def fit(self, texts, y, texts_valid=None, y_valid=None):
        self.vocab = vocabulaires(texts, min_freq=2, max_vocab=30000)

        train_ds = TextDatasetRNN(texts, y, self.vocab, self.max_len)

        y_arr = np.array(y, dtype=int)
        class_counts = np.bincount(y_arr, minlength=2)

        # sampler pour mieux exposer la classe minoritaire
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[y_arr]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )

        train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        valid_dl = None
        if texts_valid is not None and y_valid is not None:
            valid_ds = TextDatasetRNN(texts_valid, y_valid, self.vocab, self.max_len)
            valid_dl = DataLoader(
                valid_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )

        self.model = LSTMClassifier(
            vocab_size=len(self.vocab),
            emb_dim=self.emb_dim,
            hidden_dim=self.hidden_dim
        ).to(DEVICE)

        pos_count = max(int((y_arr == 1).sum()), 1)
        neg_count = max(int((y_arr == 0).sum()), 1)
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float, device=DEVICE)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-2
        )

        if self.monitor == "f1":
            best_score = -np.inf
        else:
            best_score = np.inf

        best_state = None
        patience_left = self.patience

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0.0

            for batch in train_dl:
                input_ids = batch["input_ids"].to(DEVICE)
                lengths = batch["lengths"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                optimizer.zero_grad()
                logits = self.model(input_ids, lengths)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_train_loss += loss.item()

            train_loss = total_train_loss / max(len(train_dl), 1)
            msg = f"[RNN] epoch {epoch+1}/{self.epochs} - train_loss={train_loss:.4f}"

            if valid_dl is not None:
                val_loss, y_true, y_prob = self._eval_loader(valid_dl, criterion)
                y_pred = (y_prob >= 0.5).astype(int)
                val_f1 = f1_score(y_true, y_pred)

                msg += f" - val_loss={val_loss:.4f} - val_f1={val_f1:.4f}"

                if self.monitor == "f1":
                    improved = val_f1 > best_score + self.min_delta
                    current_score = val_f1
                else:
                    improved = val_loss < best_score - self.min_delta
                    current_score = val_loss

                if improved:
                    best_score = current_score
                    patience_left = self.patience
                    best_state = {
                        "state_dict": clone_state_dict(self.model.state_dict()),
                        "best_threshold": 0.5
                    }
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(msg)
                        print("[RNN] early stopping")
                        break

            print(msg)

        if best_state is not None:
            self.model.load_state_dict(best_state["state_dict"])
            self.best_threshold = best_state["best_threshold"]

        return self

    def _eval_loader(self, dl, criterion):
        self.model.eval()
        probs = []
        labels_all = []
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in dl:
                input_ids = batch["input_ids"].to(DEVICE)
                lengths = batch["lengths"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                logits = self.model(input_ids, lengths)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

                p = torch.sigmoid(logits).cpu().numpy()
                probs.extend(p)
                labels_all.extend(labels.cpu().numpy())

        val_loss = total_val_loss / max(len(dl), 1)
        return val_loss, np.array(labels_all).astype(int), np.array(probs)

    def predict_proba(self, texts):
        self.model.eval()
        ds = TextDatasetRNN(texts, [0] * len(texts), self.vocab, self.max_len)
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        probs = []
        with torch.no_grad():
            for batch in dl:
                input_ids = batch["input_ids"].to(DEVICE)
                lengths = batch["lengths"].to(DEVICE)
                logits = self.model(input_ids, lengths)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.extend(p)

        probs = np.array(probs)
        return np.vstack([1 - probs, probs]).T

    def predict(self, texts):
        proba = self.predict_proba(texts)[:, 1]
        return (proba >= self.best_threshold).astype(int)


def rnn_lstm():
    return RNNWrapper(
        max_len=48,
        emb_dim=128,
        hidden_dim=64,
        batch_size=64,
        lr=1e-4,
        epochs=50,
        patience=4,
        monitor="f1",
        min_delta=1e-4
    )


# =========================================================
# Transformer avec Trainer
# =========================================================

def hf_compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
    
class TransformerWrapper:
    def __init__(
        self,
        model_name="camembert_base",
        max_len=64,
        batch_size=4,
        lr=1.5e-5,
        epochs=6,
        output_dir="./results_transformer"
    ):
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.output_dir = output_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )

        self.trainer = None

    def _tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_len
        )

    def fit(self, texts, y, texts_valid=None, y_valid=None):
        train_ds = HFDataset.from_dict({
            "text": list(texts),
            "labels": list(map(int, y))
        })
        train_ds = train_ds.map(
            self._tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        eval_ds = None
        if texts_valid is not None and y_valid is not None:
            eval_ds = HFDataset.from_dict({
                "text": list(texts_valid),
                "labels": list(map(int, y_valid))
            })
            eval_ds = eval_ds.map(
                self._tokenize_function,
                batched=True,
                remove_columns=["text"]
            )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        y_arr = np.array(list(map(int, y)))
        class_counts = np.bincount(y_arr, minlength=2)

        # poids équilibrés + léger bonus pour la classe 1
        class_weights = class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1))
        class_weights = class_weights.astype(np.float32)
        class_weights[1] *= 1.20
        class_weights = torch.tensor(class_weights, dtype=torch.float)

        steps_per_epoch = max(1, len(train_ds) // self.batch_size)
        warmup_steps = max(1, int(steps_per_epoch * self.epochs * 0.1))

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            weight_decay=0.01,
            # warmup_steps=warmup_steps,
            lr_scheduler_type="linear",
            logging_strategy="epoch",
            eval_strategy="epoch" if eval_ds is not None else "no",
            save_strategy="epoch" if eval_ds is not None else "no",
            load_best_model_at_end=True if eval_ds is not None else False,
            metric_for_best_model="f1",
            greater_is_better=True if eval_ds is not None else None,
            report_to="none",
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            fp16=False,
            bf16=False,
            remove_unused_columns=True,
            save_total_limit=1,
            seed=RANDOM_STATE
            # label_smoothing_factor=0.05
        )

        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

        self.trainer = WeightedTrainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=hf_compute_metrics,
            callbacks=callbacks,
            class_weights=class_weights
        )

        self.trainer.train()
        self.model = self.trainer.model
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        return self

    def predict_proba(self, texts):
        ds = HFDataset.from_dict({"text": list(texts)})
        ds = ds.map(
            self._tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        if self.trainer is not None:
            preds = self.trainer.predict(ds)
            logits = preds.predictions
            probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
            return probs

        self.model.eval()

        local_device = torch.device(DEVICE)
        self.model.to(local_device)

        enc = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        probs = []
        bs = self.batch_size

        with torch.no_grad():
            n = enc["input_ids"].size(0)
            for i in range(0, n, bs):
                batch = {k: v[i:i+bs].to(local_device) for k, v in enc.items()}
                outputs = self.model(**batch)
                p = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                probs.append(p)

        return np.vstack(probs)

    def predict(self, texts):
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)


def build_transformer():
    return TransformerWrapper(
        model_name="camembert-base",
        max_len=64,
        batch_size=4,
        lr=1.5e-5,
        epochs=6,
        output_dir="./results_transformer"
    )

# =========================================================
# Eval
# =========================================================

def evaluate_single_experiment(exp_name, model, X_train, y_train, X_valid, y_valid):
    try:
        model.fit(X_train, y_train, X_valid, y_valid)
    except TypeError:
        model.fit(X_train, y_train)

    pred_valid = model.predict(X_valid)
    proba_valid = model.predict_proba(X_valid)[:, 1]

    metrics = compute_metrics(y_valid, pred_valid, proba_valid)

    print("\n" + "=" * 50)
    print(exp_name)
    print("=" * 50)
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"F1-score  : {metrics['f1']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"Log-loss  : {metrics['log_loss']:.4f}")
    print("Confusion matrix :")
    print(metrics["confusion_matrix"])

    return {
        "name": exp_name,
        "model": model,
        **metrics
    }


# =========================================================
# Sauvegarde / chargement
# =========================================================

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def copy_any(src, dst):
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def get_sklearn_path(model_root, name):
    return os.path.join(model_root, f"{name}.joblib")


def save_sklearn_model(model, model_root, name):
    os.makedirs(model_root, exist_ok=True)
    path = get_sklearn_path(model_root, name)
    dump(model, path)
    return path


def load_sklearn_model(model_root, name):
    path = get_sklearn_path(model_root, name)
    if os.path.exists(path):
        return load(path), path
    return None, path


def get_torch_path(model_root, name):
    return os.path.join(model_root, f"{name}.pt")


def save_torch_checkpoint(checkpoint, model_root, name):
    os.makedirs(model_root, exist_ok=True)
    path = get_torch_path(model_root, name)
    torch.save(checkpoint, path)
    return path


def load_torch_checkpoint(model_root, name):
    path = get_torch_path(model_root, name)
    if os.path.exists(path):
        return torch.load(path, map_location=DEVICE), path
    return None, path


def get_transformer_dir(model_root, name):
    return os.path.join(model_root, name)


def save_transformer_model(model, tokenizer, meta, model_root, name):
    save_dir = get_transformer_dir(model_root, name)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    save_json(meta, os.path.join(save_dir, "meta.json"))
    return save_dir


def load_transformer_model(model_root, name):
    save_dir = get_transformer_dir(model_root, name)
    if not os.path.exists(save_dir):
        return None, None, None, save_dir

    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(DEVICE)

    meta_path = os.path.join(save_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return model, tokenizer, meta, save_dir


# =========================================================
# Train or load
# =========================================================

def train_or_load_sklearn(model_root, model_name, builder, X_train, y_train):
    model, path = load_sklearn_model(model_root, model_name)
    if model is not None:
        print(f"[LOAD] {path}")
        return model, path

    print(f"[TRAIN] {model_name}")
    model = builder()
    model.fit(X_train, y_train)
    path = save_sklearn_model(model, model_root, model_name)
    print(f"[SAVE] {path}")
    return model, path


def train_or_load_w2v(model_root, model_name, builder, X_train, y_train,X_valid=None,y_valid=None):
    model, path = load_sklearn_model(model_root, model_name)
    if model is not None:
        print(f"[LOAD] {path}")
        return model, path

    print(f"[TRAIN] {model_name}")
    model = builder()
    model.fit(X_train, y_train,X_valid,y_valid)
    path = save_sklearn_model(model, model_root, model_name)
    print(f"[SAVE] {path}")
    return model, path


def train_or_load_rnn(model_root, model_name, builder, X_train, y_train, X_valid=None, y_valid=None):
    ckpt, path = load_torch_checkpoint(model_root, model_name)
    if ckpt is not None:
        print(f"[LOAD] {path}")
        wrapper = builder()
        wrapper.vocab = ckpt["vocab"]
        wrapper.max_len = ckpt["max_len"]
        wrapper.emb_dim = ckpt["emb_dim"]
        wrapper.hidden_dim = ckpt["hidden_dim"]
        wrapper.best_threshold = ckpt.get("best_threshold", 0.5)
        wrapper.model = LSTMClassifier(
            vocab_size=ckpt["vocab_size"],
            emb_dim=ckpt["emb_dim"],
            hidden_dim=ckpt["hidden_dim"]
        ).to(DEVICE)
        wrapper.model.load_state_dict(ckpt["state_dict"])
        return wrapper, path

    print(f"[TRAIN] {model_name}")
    wrapper = builder()
    wrapper.fit(X_train, y_train, X_valid, y_valid)

    ckpt = {
        "state_dict": wrapper.model.state_dict(),
        "vocab": wrapper.vocab,
        "vocab_size": len(wrapper.vocab),
        "emb_dim": wrapper.emb_dim,
        "hidden_dim": wrapper.hidden_dim,
        "max_len": wrapper.max_len,
        "best_threshold": wrapper.best_threshold
    }
    path = save_torch_checkpoint(ckpt, model_root, model_name)
    print(f"[SAVE] {path}")
    return wrapper, path


def train_or_load_transformer(model_root, model_name, builder, X_train, y_train, X_valid=None, y_valid=None):
    model, tokenizer, meta, save_dir = load_transformer_model(model_root, model_name)
    if model is not None:
        print(f"[LOAD] {save_dir}")
        wrapper = builder()
        wrapper.model = model
        wrapper.tokenizer = tokenizer
        wrapper.trainer = None

        if "max_len" in meta:
            wrapper.max_len = meta["max_len"]
        if "model_name" in meta:
            wrapper.model_name = meta["model_name"]

        return wrapper, save_dir

    print(f"[TRAIN] {model_name}")
    wrapper = builder()
    wrapper.fit(X_train, y_train, X_valid, y_valid)

    meta = {
        "model_name": wrapper.model_name,
        "max_len": wrapper.max_len
    }
    save_dir = save_transformer_model(wrapper.model, wrapper.tokenizer, meta, model_root, model_name)
    print(f"[SAVE] {save_dir}")
    return wrapper, save_dir

