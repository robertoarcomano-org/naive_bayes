# -*- coding: utf-8 -*-
"""
Costruisce il dataset di addestramento (testi + categorie) a partire dal
testo estratto dai 166 numeri storici de "la Piazza", e lo salva in un
file JSON esterno (dataset_la_piazza.json).

Approccio:
 1. Si legge il testo grezzo (estratto con pdftotext) di ogni numero.
 2. Il testo viene diviso in paragrafi e questi vengono raggruppati in
    "pseudo-articoli" di lunghezza ragionevole (300-1500 caratteri circa),
    scartando righe di rumore (pubblicita', numeri di telefono, intestazioni).
 3. Ad ogni pseudo-articolo vengono assegnate una o piu' categorie tramite
    un dizionario di parole chiave (weak supervision), perche' etichettare
    a mano migliaia di articoli su 20 anni di giornale non e' fattibile.
 4. Gli pseudo-articoli senza nessuna categoria riconosciuta vengono
    scartati; il risultato viene sotto-campionato per restare in un
    dataset maneggevole e viene scritto su JSON.
"""

import json
import random
import re
from pathlib import Path

random.seed(42)

TXT_DIR = Path("/home/claude/pdfs/txt")
OUTPUT_JSON = Path("/home/claude/dataset_la_piazza.json")

MAX_CHUNKS_PER_ISSUE = 6       # limite per numero, per non sbilanciare il dataset
MIN_CHUNK_LEN = 300
MAX_CHUNK_LEN = 1500

# -----------------------------------------------------------------
# Dizionario di parole chiave per categoria (in minuscolo)
# -----------------------------------------------------------------

CATEGORIE_KEYWORDS = {
    "Maltempo e dissesto idrogeologico": [
        "maltempo", "alluvione", "frana", "frane", "smottamento", "nubifragio",
        "dissesto idrogeologico", "esondazione", "temporale", "gelo",
        "stato di emergenza", "calamita naturale", "ondata di maltempo",
    ],
    "Viabilita e infrastrutture": [
        "viabilita", "strada provinciale", "strada comunale", "sp89", "ss106",
        "asfalto", "messa in sicurezza", "ponte", "cantiere", "lavori stradali",
        "circolazione", "traffico", "infrastrutture stradali", "strada interpoderale",
    ],
    "Politica e amministrazione locale": [
        "consiglio comunale", "giunta comunale", "sindaco", "delibera", "assessore",
        "elezioni", "referendum", "amministrazione comunale", "consiglio regionale",
        "regione basilicata", "candidato", "comune di roccanova", "prefettura",
        "consigliere", "opposizione", "maggioranza",
    ],
    "Economia e fondi regionali": [
        "fondi", "finanziamento", "bando pubblico", "milioni di euro", "investimento",
        "sviluppo economico", "occupazione", "impresa", "pnrr", "contributo economico",
        "stanziamento", "risorse regionali", "fondo perduto", "bilancio",
    ],
    "Sanita": [
        "ospedale", "sanita", "asl", "ambulatorio", "guardia medica", "medico",
        "presidio ospedaliero", "emergenza sanitaria", "vaccino", "pronto soccorso",
        "medici di base", "reparto",
    ],
    "Istruzione e giovani": [
        "scuola", "istituto scolastico", "studenti", "universita", "borsa di studio",
        "laurea", "docenti", "alunni", "asilo", "scuola materna", "scuola media",
        "liceo", "istruzione", "studentesse", "professoresse", "professori",
    ],
    "Demografia e spopolamento": [
        "spopolamento", "popolazione", "abitanti", "residenti", "istat",
        "denatalita", "emigrazione", "natalita", "censimento", "comuni lucani",
        "andamento demografico", "nati", "morti", "matrimoni", "immigrati", "emigrati",
    ],
    "Cultura e tradizioni": [
        "cultura", "tradizione", "dialetto", "libro", "presentazione del libro",
        "mostra", "storia locale", "patrimonio culturale", "museo", "carlo levi",
        "poesia", "teatro", "convegno", "conferenza", "presentazione",
    ],
    "Cronaca e comunita": [
        "compleanno", "anniversario", "necrologio", "e morto", "e deceduto",
        "lutto", "festeggiamenti", "cerimonia", "premiazione", "ricorrenza",
        "auguri", "roccanovesi nel mondo", "roccanova nel mondo",
    ],
    "Sport": [
        "calcio", "campionato", "squadra", "torneo", "polisportiva", "partita",
        "gol", "playoff", "play off", "pallavolo", "sportivo", "allenatore",
        "sportiva", "classifica", "girone",
    ],
    "Ambiente ed energia": [
        "ambiente", "energia rinnovabile", "fotovoltaico", "raccolta differenziata",
        "rifiuti", "risparmio energetico", "sostenibilita", "impianto eolico",
        "riciclo", "fonti rinnovabili", "metanizzati",
    ],
    "Servizi pubblici acqua": [
        "acqua potabile", "acquedotto", "rete idrica", "bolletta", "potabilizzatore",
        "servizio idrico", "tariffa idrica", "acquedotto lucano",
    ],
    "Religione e feste patronali": [
        "festa patronale", "san rocco", "parrocchia", "processione", "messa",
        "chiesa", "sacerdote", "vescovo", "pellegrinaggio", "madonna",
        "corpus domini", "parroco", "religiosa", "festa religiosa",
    ],
    "Agricoltura ed enogastronomia": [
        "agricoltura", "vino", "cantina", "vigneto", "olio", "agriturismo",
        "dop", "igp", "prodotti tipici", "enogastronomia", "aglianico",
        "produttori agricoli", "vendemmia", "oleificio", "cantine",
    ],
}

# righe/rumore da scartare nei paragrafi
NOISE_PATTERNS = [
    re.compile(r"^\s*tel\.?\s*\d", re.I),
    re.compile(r"^\s*cell\.?\s*\d", re.I),
    re.compile(r"^\s*via\s+\w+.*\d", re.I),
    re.compile(r"anno\s+[ivxlcdm]+", re.I),
    re.compile(r"^\s*p\.?\s*iva", re.I),
    re.compile(r"e-?mail", re.I),
    re.compile(r"^\s*www\.", re.I),
]


def normalizza(testo: str) -> str:
    testo = testo.replace("’", "'").replace("'", "'")
    testo = re.sub(r"[àáâã]", "a", testo)
    testo = re.sub(r"[èéêë]", "e", testo)
    testo = re.sub(r"[ìíîï]", "i", testo)
    testo = re.sub(r"[òóôõ]", "o", testo)
    testo = re.sub(r"[ùúûü]", "u", testo)
    return testo.lower()


def e_rumore(paragrafo: str) -> bool:
    if len(paragrafo.strip()) < 40:
        return True
    for pattern in NOISE_PATTERNS:
        if pattern.search(paragrafo):
            return True
    lettere = sum(c.isalpha() for c in paragrafo)
    if lettere / max(len(paragrafo), 1) < 0.55:
        return True
    return False


def segmenta_in_chunk(testo_grezzo: str):
    """Divide il testo di un intero numero in pseudo-articoli."""
    paragrafi = re.split(r"\n\s*\n", testo_grezzo)
    paragrafi_puliti = []
    for p in paragrafi:
        p = re.sub(r"\s+", " ", p).strip()
        if not e_rumore(p):
            paragrafi_puliti.append(p)

    chunk_attuale = ""
    chunks = []
    for p in paragrafi_puliti:
        if len(chunk_attuale) + len(p) < MAX_CHUNK_LEN:
            chunk_attuale = (chunk_attuale + " " + p).strip()
        else:
            if len(chunk_attuale) >= MIN_CHUNK_LEN:
                chunks.append(chunk_attuale)
            chunk_attuale = p
    if len(chunk_attuale) >= MIN_CHUNK_LEN:
        chunks.append(chunk_attuale)
    return chunks


def assegna_categorie(chunk: str):
    chunk_norm = normalizza(chunk)
    etichette = []
    for categoria, keywords in CATEGORIE_KEYWORDS.items():
        for kw in keywords:
            if kw in chunk_norm:
                etichette.append(categoria)
                break
    return etichette


def main():
    dataset = []
    file_txt = sorted(TXT_DIR.glob("*.txt"))
    print(f"Trovati {len(file_txt)} file di testo da elaborare...")

    for path in file_txt:
        testo = path.read_text(encoding="utf-8", errors="ignore")
        chunks = segmenta_in_chunk(testo)

        candidati = []
        for chunk in chunks:
            etichette = assegna_categorie(chunk)
            if etichette:
                candidati.append((chunk, etichette))

        # sotto-campiona per numero, per non sbilanciare troppo il dataset
        random.shuffle(candidati)
        candidati = candidati[:MAX_CHUNKS_PER_ISSUE]

        for chunk, etichette in candidati:
            dataset.append({
                "fonte": path.stem,
                "testo": chunk,
                "categorie": sorted(set(etichette)),
            })

    print(f"Pseudo-articoli etichettati raccolti: {len(dataset)}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Dataset salvato in: {OUTPUT_JSON}")

    # statistiche rapide
    conteggio = {}
    for item in dataset:
        for cat in item["categorie"]:
            conteggio[cat] = conteggio.get(cat, 0) + 1
    print("\nDistribuzione categorie:")
    for cat, n in sorted(conteggio.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
