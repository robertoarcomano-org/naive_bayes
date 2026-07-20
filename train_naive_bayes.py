# -*- coding: utf-8 -*-
"""
Addestra un classificatore Naive Bayes multi-etichetta per riconoscere
le categorie degli articoli del giornale "la Piazza" (Roccanova),
a partire da un dataset esterno in formato JSON.

Il file "dataset_la_piazza.json" deve trovarsi nella stessa cartella
dello script (o il percorso va aggiornato in DATASET_PATH) e deve avere
questa struttura:

[
  {
    "fonte": "nome_del_numero_di_origine",
    "testo": "testo dell'articolo/pseudo-articolo...",
    "categorie": ["Categoria 1", "Categoria 2"]
  },
  ...
]

Pipeline:
  1. Caricamento del dataset dal JSON
  2. Vettorizzazione del testo con TF-IDF
  3. Binarizzazione delle etichette multi-label (MultiLabelBinarizer)
  4. Classificatore OneVsRest con MultinomialNB (uno per ogni categoria)
  5. Valutazione con classification_report

Richiede: scikit-learn (pip install scikit-learn --break-system-packages)
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATASET_PATH = Path(__file__).parent / "dataset_la_piazza.json"

# Stopword italiane essenziali (elenco ridotto, sufficiente per un TF-IDF
# ragionevole senza dover installare pacchetti aggiuntivi tipo nltk)
STOPWORDS_IT = [
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "a", "da",
    "in", "con", "su", "per", "tra", "fra", "e", "ed", "o", "ma", "che",
    "chi", "cui", "non", "come", "anche", "piu", "meno", "ci", "si", "ne",
    "questo", "questa", "questi", "queste", "quello", "quella", "quelli",
    "quelle", "del", "della", "dello", "dei", "degli", "delle", "al", "allo",
    "alla", "ai", "agli", "alle", "dal", "dallo", "dalla", "dai", "dagli",
    "dalle", "nel", "nello", "nella", "nei", "negli", "nelle", "sul", "sullo",
    "sulla", "sui", "sugli", "sulle", "sono", "ha", "hanno", "ho", "abbiamo",
    "avete", "sara", "saranno", "essere", "fatto", "fare", "molto", "poco",
    "tutto", "tutti", "tutte", "loro", "suo", "sua", "suoi", "sue", "mio",
    "mia", "miei", "mie",
]


def carica_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        dati = json.load(f)
    testi = [item["testo"] for item in dati]
    categorie = [item["categorie"] for item in dati]
    return testi, categorie


def main():
    # -----------------------------------------------------------------
    # 1. CARICAMENTO DEL DATASET DAL FILE JSON ESTERNO
    # -----------------------------------------------------------------
    testi, categorie = carica_dataset(DATASET_PATH)
    print(f"Caricati {len(testi)} esempi dal file {DATASET_PATH.name}")

    # -----------------------------------------------------------------
    # 2. PREPARAZIONE DELLE ETICHETTE MULTI-LABEL
    # -----------------------------------------------------------------
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(categorie)
    print(f"\nCategorie riconosciute dal modello ({len(mlb.classes_)}):")
    for cat in mlb.classes_:
        print(f"  - {cat}")

    X_train, X_test, y_train, y_test = train_test_split(
        testi, y, test_size=0.2, random_state=42
    )

    # -----------------------------------------------------------------
    # 3. VETTORIZZAZIONE TF-IDF
    # -----------------------------------------------------------------
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=STOPWORDS_IT,
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=2,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # -----------------------------------------------------------------
    # 4. ADDESTRAMENTO DEL MODELLO NAIVE BAYES MULTI-LABEL
    # -----------------------------------------------------------------
    modello = OneVsRestClassifier(MultinomialNB())
    modello.fit(X_train_tfidf, y_train)

    # -----------------------------------------------------------------
    # 5. VALUTAZIONE
    # -----------------------------------------------------------------
    y_pred = modello.predict(X_test_tfidf)

    print("\nReport di classificazione sul test set:")
    print(classification_report(
        y_test, y_pred, target_names=mlb.classes_, zero_division=0
    ))

    # -----------------------------------------------------------------
    # 6. FUNZIONE DI UTILITA' PER PREDIRE LE CATEGORIE DI UN NUOVO TESTO
    # -----------------------------------------------------------------
    def predici_categorie(testo_nuovo):
        vettore = vectorizer.transform([testo_nuovo])
        predizione = modello.predict(vettore)
        etichette = mlb.inverse_transform(predizione)
        return etichette[0]

    esempio = (
        "Una nuova frana ha colpito la provinciale causando ingenti danni "
        "e la Regione ha stanziato fondi per l'emergenza."
    )
    print("\nEsempio di predizione su un nuovo testo:")
    print(f"Testo: {esempio}")
    print(f"Categorie predette: {predici_categorie(esempio)}")


if __name__ == "__main__":
    main()
