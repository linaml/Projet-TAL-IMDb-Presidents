import os
import re
import html
import json
import shutil
import random
import joblib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss,
)
from sklearn.model_selection import cross_val_score, train_test_split,StratifiedKFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# =========================================================
# Configuration globale
# =========================================================

RANDOM_STATE = 42
MAX_WORDS = 20000
MAX_LEN = 200
EMBED_DIM = 64
BATCH_SIZE = 16
EPOCHS = 5

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# =========================================================
# Chemins projet
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "Models" / "Movies"
BEST_MODELS_DIR = MODELS_DIR / "BEST"
BEST_GLOBAL_DIR = MODELS_DIR / "THE BEST"
REPORTS_DIR = PROJECT_ROOT / "Codes" / "Movies" / "Reports"
CONFUSION_DIR = REPORTS_DIR / "confusion_matrices"


def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Chargement des données
# =========================================================

def load_movies(path2data):
    texts = []
    labels = []

    classes = sorted(
        [d for d in os.listdir(path2data) if os.path.isdir(os.path.join(path2data, d))]
    )

    label2id = {cl: i for i, cl in enumerate(classes)}
    id2label = {i: cl for cl, i in label2id.items()}

    for cl in classes:
        class_dir = os.path.join(path2data, cl)
        for fname in sorted(os.listdir(class_dir)):
            fpath = os.path.join(class_dir, fname)
            if os.path.isfile(fpath):
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                texts.append(txt)
                labels.append(label2id[cl])

    return texts, labels, label2id, id2label


def load_test_file(test_file):
    texts = []
    with open(test_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts


def inspect_test_file(test_file: str, n_lines: int = 5) -> None:
    with open(test_file, "r", encoding="utf-8", errors="ignore") as f:
        for i in range(n_lines):
            line = f.readline()
            if not line:
                break
            print(f"[Ligne {i}] {repr(line[:300])}")


def split_train_val_test(texts, labels, test_size=0.2, val_size=0.2, random_state=42):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================================================
# Nettoyage
# =========================================================

def _normalize_html(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_raw_minimal(text: str) -> str:
    text = _normalize_html(text)
    text = _normalize_spaces(text)
    return text


def clean_light(text: str) -> str:
    text = text.lower()
    text = _normalize_html(text)
    text = re.sub(r"[^a-z0-9\s.,!?;:'\"()\-]", " ", text)
    text = _normalize_spaces(text)
    return text


def clean_no_punct(text: str) -> str:
    text = text.lower()
    text = _normalize_html(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = _normalize_spaces(text)
    return text


def clean_keep_negation(text: str) -> str:
    text = text.lower()
    text = _normalize_html(text)
    text = re.sub(r"n['’]t\b", " not", text)
    text = re.sub(r"['’]re\b", " are", text)
    text = re.sub(r"['’]s\b", " is", text)
    text = re.sub(r"['’]m\b", " am", text)
    text = re.sub(r"['’]ll\b", " will", text)
    text = re.sub(r"['’]ve\b", " have", text)
    text = re.sub(r"['’]d\b", " would", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = _normalize_spaces(text)
    return text


CLEANINGS: Dict[str, Callable] = {
    "raw_minimal": clean_raw_minimal,
    "light": clean_light,
    "keep_negation": clean_keep_negation,
    "no_punct": clean_no_punct
}


# =========================================================
# Stopwords
# =========================================================

def get_custom_stopwords():
    keep_words = {"no", "not", "nor", "never"}
    return list(set(ENGLISH_STOP_WORDS) - keep_words)


def apply_cleaning(texts: List[str], clean_fn) -> List[str]:
    return [clean_fn(t) for t in texts]


def apply_cleaning_with_stopwords(texts: List[str], clean_fn, stopwords=None) -> List[str]:
    cleaned = []
    stopwords_set = set(stopwords) if stopwords is not None else None
    for t in texts:
        t = clean_fn(t)
        if stopwords_set is not None:
            tokens = [tok for tok in t.split() if tok and tok not in stopwords_set]
            t = " ".join(tokens)
        cleaned.append(t)
    return cleaned


# =========================================================
# Export final
# =========================================================

def infer_positive_negative_mapping(label2id):
    pos_candidates = {"pos", "positive", "positif", "p"}
    neg_candidates = {"neg", "negative", "negatif", "n"}

    pos_id = None
    neg_id = None

    for label_name, idx in label2id.items():
        lname = label_name.lower()
        if lname in pos_candidates:
            pos_id = idx
        elif lname in neg_candidates:
            neg_id = idx

    return pos_id, neg_id


def convert_preds_to_PN(preds, pos_id, neg_id):
    out = []
    for p in preds:
        if int(p) == int(pos_id):
            out.append("P")
        elif int(p) == int(neg_id):
            out.append("N")
        else:
            raise ValueError(f"Label inconnu: {p}")
    return out


def save_submission_csv(pred_letters, out_csv="submission.csv", header=False):
    df = pd.DataFrame(pred_letters)
    df.to_csv(out_csv, index=False, header=header)
    print(f"CSV sauvegardé dans : {out_csv}")


# =========================================================
# Dataclass résultats
# =========================================================

@dataclass
class ExperimentResult:
    family: str
    model_name: str
    clean_name: str
    stopwords_name: str
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    train_f1: float
    val_f1: float
    test_f1: float
    train_loss: Optional[float]
    val_loss: Optional[float]
    test_loss: Optional[float]
    cv_f1_mean: Optional[float]
    cv_f1_std: Optional[float]
    model_path: str
    report_path: str
    confusion_matrix_path: str


# =========================================================
# Nommage chemins
# =========================================================

def build_experiment_id(family: str, model_name: str, clean_name: str, stopwords_name: str):
    return f"{family}__{model_name}__{clean_name}__{stopwords_name}"


def sklearn_model_path(family: str, model_name: str, clean_name: str, stopwords_name: str):
    exp_id = build_experiment_id(family, model_name, clean_name, stopwords_name)
    return str(MODELS_DIR / f"{exp_id}.joblib")


def keras_model_path(family: str, model_name: str, clean_name: str, stopwords_name: str):
    exp_id = build_experiment_id(family, model_name, clean_name, stopwords_name)
    return str(MODELS_DIR / f"{exp_id}.keras")


def transformer_model_dir(family: str, model_name: str, clean_name: str, stopwords_name: str):
    exp_id = build_experiment_id(family, model_name, clean_name, stopwords_name)
    return str(MODELS_DIR / exp_id)


def report_path(family: str, model_name: str, clean_name: str, stopwords_name: str):
    exp_id = build_experiment_id(family, model_name, clean_name, stopwords_name)
    return str(REPORTS_DIR / f"{exp_id}.json")


def confusion_path(family: str, model_name: str, clean_name: str, stopwords_name: str):
    exp_id = build_experiment_id(family, model_name, clean_name, stopwords_name)
    return str(CONFUSION_DIR / f"{exp_id}.png")


def history_plot_path(family: str, model_name: str, clean_name: str, stopwords_name: str):
    exp_id = build_experiment_id(family, model_name, clean_name, stopwords_name)
    return str(REPORTS_DIR / exp_id)


# =========================================================
# Outils évaluation
# =========================================================

def maybe_compute_log_loss(model, X, y):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return float(log_loss(y, probs))
    return None


def save_confusion_matrix(y_true, y_pred, labels: List[str], out_path: str, title: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_report_json(report_dict, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)


def save_keras_history_plot(history, out_path_prefix):
    Path(out_path_prefix).parent.mkdir(parents=True, exist_ok=True)
    hist = history.history

    if "loss" in hist:
        plt.figure(figsize=(6, 4))
        plt.plot(hist["loss"], label="train_loss")
        if "val_loss" in hist:
            plt.plot(hist["val_loss"], label="val_loss")
        plt.legend()
        plt.title("Loss")
        plt.tight_layout()
        plt.savefig(f"{out_path_prefix}_loss.png", dpi=150)
        plt.close()

    if "accuracy" in hist:
        plt.figure(figsize=(6, 4))
        plt.plot(hist["accuracy"], label="train_acc")
        if "val_accuracy" in hist:
            plt.plot(hist["val_accuracy"], label="val_acc")
        plt.legend()
        plt.title("Accuracy")
        plt.tight_layout()
        plt.savefig(f"{out_path_prefix}_accuracy.png", dpi=150)
        plt.close()


def save_results_table(results, out_csv=None):
    ensure_dirs()

    if out_csv is None:
        out_csv = REPORTS_DIR / "all_results.csv"

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([asdict(r) for r in results])
    df["gap_f1"] = df["train_f1"] - df["val_f1"]
    df = df.sort_values(by=["val_f1", "test_f1", "gap_f1"], ascending=[False, False, True])
    df.to_csv(out_csv, index=False)
    return df


def copy_best_models(df: pd.DataFrame):
    ensure_dirs()

    ranking_df = df.copy()
    ranking_df["gap_f1_abs"] = (ranking_df["train_f1"] - ranking_df["val_f1"]).abs()

    for family in ranking_df["family"].unique():
        sub = ranking_df[ranking_df["family"] == family].sort_values(
            by=["val_f1", "test_f1", "gap_f1_abs"],
            ascending=[False, False, True]
        )
        best_row = sub.iloc[0]
        src = best_row["model_path"]
        model_name = best_row["model_name"]

        if os.path.isdir(src):
            dst = BEST_MODELS_DIR / f"{family}__{model_name}__BEST"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            ext = os.path.splitext(src)[1]
            dst = BEST_MODELS_DIR / f"{family}__{model_name}__BEST{ext}"
            shutil.copy2(src, dst)

        with open(BEST_MODELS_DIR / f"{family}__{model_name}__BEST.json", "w", encoding="utf-8") as f:
            json.dump(best_row.to_dict(), f, indent=2)

    best_global = ranking_df.sort_values(
        by=["val_f1", "test_f1", "gap_f1_abs"],
        ascending=[False, False, True]
    ).iloc[0]

    src = best_global["model_path"]
    family = best_global["family"]
    model_name = best_global["model_name"]

    if os.path.isdir(src):
        dst = BEST_GLOBAL_DIR / f"{family}__{model_name}__THE_BEST"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        ext = os.path.splitext(src)[1]
        dst = BEST_GLOBAL_DIR / f"{family}__{model_name}__THE_BEST{ext}"
        shutil.copy2(src, dst)

    with open(BEST_GLOBAL_DIR / "best_summary.json", "w", encoding="utf-8") as f:
        json.dump(best_global.to_dict(), f, indent=2)

# =========================================================
# Sklearn
# =========================================================

def build_sklearn_pipeline(model_name, stopwords, tuned=False):
    model_key = model_name.strip().lower()

    aliases = {
        "linearsvc": "linearsvc",
        "linear_svc": "linearsvc",
        "svc": "linearsvc",
        "logreg": "logreg",
        "logisticregression": "logreg",
        "multinomialnb": "multinomialnb",
        "nb": "multinomialnb",
    }
    model_key = aliases.get(model_key, model_key)

    if tuned:
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            min_df=5,
            max_df=0.85,
            sublinear_tf=True,
            stop_words=stopwords,
        )

        if model_key == "linearsvc":
            clf = LinearSVC(C=0.5)
        elif model_key == "logreg":
            clf = LogisticRegression(
                max_iter=3000,
                C=0.5,
                solver="liblinear"
            )
        elif model_key == "multinomialnb":
            clf = MultinomialNB(alpha=0.8)
        else:
            raise ValueError(f"Modèle inconnu: {model_name}")
    else:
        tfidf = TfidfVectorizer(
            ngram_range=(1, 1),
            max_features=5000,
            min_df=5,
            max_df=0.90,
            sublinear_tf=True,
            stop_words=stopwords,
        )

        if model_key == "linearsvc":
            clf = LinearSVC(C=1.0)
        elif model_key == "logreg":
            clf = LogisticRegression(max_iter=3000, solver="liblinear", C=1.0)
        elif model_key == "multinomialnb":
            clf = MultinomialNB(alpha=1.0)
        else:
            raise ValueError(f"Modèle inconnu: {model_name}")

    return Pipeline([
        ("tfidf", tfidf),
        ("clf", clf),
    ])

def add_experiment_tag(clean_name: str, stopwords_name: str, tuned: bool = False):
    tag = "default"
    if tuned:
        tag = "tuned"
    return f"{clean_name}__{stopwords_name}__{tag}"


def train_or_load_sklearn_model(family, model_name, clean_name, stopwords_name, pipeline, X_train, y_train):
    ensure_dirs()
    fpath = sklearn_model_path(family, model_name, clean_name, stopwords_name)
    Path(fpath).parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(fpath):
        print(f"[LOAD] {fpath}")
        model = joblib.load(fpath)
    else:
        print(f"[TRAIN] {fpath}")
        model = pipeline.fit(X_train, y_train)
        joblib.dump(model, fpath)
        print(f"[SAVE] {fpath}")

    return model, fpath


def evaluate_sklearn_model(
    family,
    model_name,
    clean_name,
    stopwords_name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    label_names,
    do_cv=True,
    cv=5,
):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred, average="macro")
    val_f1 = f1_score(y_val, y_val_pred, average="macro")
    test_f1 = f1_score(y_test, y_test_pred, average="macro")

    train_loss = maybe_compute_log_loss(model, X_train, y_train)
    val_loss = maybe_compute_log_loss(model, X_val, y_val)
    test_loss = maybe_compute_log_loss(model, X_test, y_test)

    cv_mean, cv_std = None, None
    if do_cv:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring="f1_macro")
        cv_mean = float(scores.mean())
        cv_std = float(scores.std())

    cm_path = confusion_path(family, model_name, clean_name, stopwords_name)
    save_confusion_matrix(
        y_test, y_test_pred, label_names, cm_path,
        title=f"{model_name} | {clean_name} | {stopwords_name}"
    )

    rep = {
        "family": family,
        "model_name": model_name,
        "clean_name": clean_name,
        "stopwords_name": stopwords_name,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "test_f1": test_f1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "cv_f1_mean": cv_mean,
        "cv_f1_std": cv_std,
        "classification_report": classification_report(
            y_test, y_test_pred, target_names=label_names, output_dict=True
        ),
        "model_path": sklearn_model_path(family, model_name, clean_name, stopwords_name),
        "confusion_matrix_path": cm_path,
    }

    rep_p = report_path(family, model_name, clean_name, stopwords_name)
    save_report_json(rep, rep_p)

    return ExperimentResult(
        family=family,
        model_name=model_name,
        clean_name=clean_name,
        stopwords_name=stopwords_name,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        test_accuracy=test_acc,
        train_f1=train_f1,
        val_f1=val_f1,
        test_f1=test_f1,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
        cv_f1_mean=cv_mean,
        cv_f1_std=cv_std,
        model_path=sklearn_model_path(family, model_name, clean_name, stopwords_name),
        report_path=rep_p,
        confusion_matrix_path=cm_path,
    )


# =========================================================
# Keras BiLSTM
# =========================================================

def build_bilstm_model(max_words=12000, max_len=150, embed_dim=48):
    model = keras.Sequential([
        keras.layers.Embedding(
            input_dim=max_words,
            output_dim=embed_dim,
            input_length=max_len,
            mask_zero=True
        ),
        keras.layers.Bidirectional(
            keras.layers.LSTM(
                24,
                dropout=0.25,
                recurrent_dropout=0.25
            )
        ),
        keras.layers.Dense(
            12,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(3e-4)
        ),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    optimizer = keras.optimizers.Adam(learning_rate=3e-4)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def prepare_lstm_tokenizer(X_train_texts, max_words=20000):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_texts)
    return tokenizer


def texts_to_padded(tokenizer, texts, max_len=200):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")


def save_tokenizer(tokenizer, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())


def load_tokenizer_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        tok_json = f.read()
    return keras.preprocessing.text.tokenizer_from_json(tok_json)


# =========================================================
# Transformer DistilBERT
# =========================================================

def build_transformer_datasets(tokenizer, X_train, y_train, X_val, y_val, X_test, y_test, max_len=256):
    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    val_ds = Dataset.from_dict({"text": X_val, "label": y_val})
    test_ds = Dataset.from_dict({"text": X_test, "label": y_test})

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)

    train_ds = train_ds.remove_columns(["text"])
    val_ds = val_ds.remove_columns(["text"])
    test_ds = test_ds.remove_columns(["text"])

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    return train_ds, val_ds, test_ds


def compute_metrics_transformer(eval_pred):
    logits, labels_ = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels_, preds)
    f1 = f1_score(labels_, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def train_or_load_transformer_model(
    family,
    model_name,
    clean_name,
    stopwords_name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
    ensure_dirs()
    model_dir = transformer_model_dir(family, model_name, clean_name, stopwords_name)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
        print(f"[LOAD] {model_dir}")
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    else:
        print(f"[TRAIN] {model_dir}")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2
        )

        train_ds, val_ds, _ = build_transformer_datasets(
            tokenizer, X_train, y_train, X_val, y_val, X_test, y_test
        )

        training_args = TrainingArguments(
            output_dir=model_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=1.5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            weight_decay=0.05,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
            seed=RANDOM_STATE,
        )

        trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_transformer,
        )

        trainer.train()
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)

    return model, tokenizer, model_dir


def evaluate_transformer_model(
    family,
    model_name,
    clean_name,
    stopwords_name,
    model,
    tokenizer,
    model_dir,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    label_names,
):
    train_ds, val_ds, test_ds = build_transformer_datasets(
        tokenizer, X_train, y_train, X_val, y_val, X_test, y_test
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_transformer,
    )

    train_pred = trainer.predict(train_ds)
    val_pred = trainer.predict(val_ds)
    test_pred = trainer.predict(test_ds)

    y_train_pred = np.argmax(train_pred.predictions, axis=1)
    y_val_pred = np.argmax(val_pred.predictions, axis=1)
    y_test_pred = np.argmax(test_pred.predictions, axis=1)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred, average="macro")
    val_f1 = f1_score(y_val, y_val_pred, average="macro")
    test_f1 = f1_score(y_test, y_test_pred, average="macro")

    train_probs = tf.nn.softmax(train_pred.predictions, axis=1).numpy()
    val_probs = tf.nn.softmax(val_pred.predictions, axis=1).numpy()
    test_probs = tf.nn.softmax(test_pred.predictions, axis=1).numpy()

    train_loss = float(log_loss(y_train, train_probs))
    val_loss = float(log_loss(y_val, val_probs))
    test_loss = float(log_loss(y_test, test_probs))

    cm_path = confusion_path(family, model_name, clean_name, stopwords_name)
    save_confusion_matrix(
        y_test,
        y_test_pred,
        label_names,
        cm_path,
        title=f"{model_name} | {clean_name} | {stopwords_name}"
    )

    rep = {
        "family": family,
        "model_name": model_name,
        "clean_name": clean_name,
        "stopwords_name": stopwords_name,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "test_f1": test_f1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "cv_f1_mean": None,
        "cv_f1_std": None,
        "classification_report": classification_report(
            y_test, y_test_pred, target_names=label_names, output_dict=True
        ),
        "model_path": model_dir,
        "confusion_matrix_path": cm_path,
    }

    rep_p = report_path(family, model_name, clean_name, stopwords_name)
    save_report_json(rep, rep_p)

    return ExperimentResult(
        family=family,
        model_name=model_name,
        clean_name=clean_name,
        stopwords_name=stopwords_name,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        test_accuracy=test_acc,
        train_f1=train_f1,
        val_f1=val_f1,
        test_f1=test_f1,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
        cv_f1_mean=None,
        cv_f1_std=None,
        model_path=model_dir,
        report_path=rep_p,
        confusion_matrix_path=cm_path,
    )


# =========================================================
# Prédiction meilleur modèle
# =========================================================
def predict_with_best_global(best_row, X_final_raw):
    family = best_row["family"]
    clean_name_raw = best_row["clean_name"]
    stopwords_name = best_row["stopwords_name"]
    model_path = best_row["model_path"]

    clean_name = clean_name_raw.replace("__tuned", "")
    clean_fn = CLEANINGS.get(clean_name, None)
    if clean_fn is None:
        raise ValueError(f"Cleaning inconnu: {clean_name_raw}")

    stopwords = None
    if stopwords_name == "custom_stopwords":
        stopwords = get_custom_stopwords()

    X_final_clean = apply_cleaning_with_stopwords(X_final_raw, clean_fn, stopwords)

    if family == "sklearn":
        model = joblib.load(model_path)
        return model.predict(X_final_clean)

    if family == "keras":
        model = load_model(model_path)
        tokenizer_path = model_path.replace(".keras", "_tokenizer.json")
        tokenizer = load_tokenizer_from_json(tokenizer_path)
        X_pad = texts_to_padded(tokenizer, X_final_clean, max_len=MAX_LEN)
        probs = model.predict(X_pad, verbose=0).ravel()
        return (probs >= 0.5).astype(int)

    if family == "transformer":
        def tokenize_function(batch):
            return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        test_ds = Dataset.from_dict({"text": X_final_clean})

        test_ds = test_ds.map(tokenize_function, batched=True)
        test_ds = test_ds.remove_columns(["text"])
        test_ds.set_format("torch")

        trainer = Trainer(model=model, processing_class=tokenizer)
        pred_output = trainer.predict(test_ds)
        return np.argmax(pred_output.predictions, axis=1)

    raise ValueError(f"Famille inconnue: {family}")