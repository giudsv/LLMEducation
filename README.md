# Valutazione Sperimentale degli LLM nella Didattica Universitaria

> **Repository ufficiale della Tesi di Laurea in Informatica** > **Università degli Studi di Salerno** > *Uso dei Large Language Models a supporto dell'insegnamento e della comprensione didattica*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()
[![Data](https://img.shields.io/badge/Data-Open_Access-green.svg)]()

## 📄 Descrizione del Progetto
Questa repository contiene il codice sorgente, i dataset (grezzi e valutati) e la pipeline di analisi sviluppati per la tesi di laurea volta a misurare l'efficacia dei **Large Language Models (LLM)** come strumenti di supporto alla didattica universitaria.

Il progetto garantisce la **totale riproducibilità** degli esperimenti: dai dati grezzi (`rawdata`), passando per la valutazione automatica (`pipeline`), fino all'analisi statistica e alla generazione dei grafici (`analysis`).

---

## 🗂 Organizzazione del Repository

Ecco come sono organizzati i file nel progetto:

```text
├── executive_summary.md       # Sintesi dei risultati principali (Score medi, Ranking)
├── README.md                  # Questo file
│
├── analysis/                  # Modulo di analisi statistica
│   ├── analyze_results_advanced.py  # Script per test statistici (Kruskal-Wallis) e grafici
│   ├── analisi_statistica_avanzata.xlsx # Report tabellare completo generato dallo script
│   │
│   └── output/                # Output generati dall'analisi
│       ├── figs/              # Grafici salvati (Boxplot, Heatmap, Scatter plot)
│       └── logs/              # AUDIT TRAIL: File JSON singoli per ogni valutazione (tracciabilità totale)
│
├── data/                      # Dataset
│   ├── risultati_*_scored.xlsx      # Dati finali valutati dall'Agente (con Score e Note)
│   │
│   └── rawdata/               # Dati grezzi (Input)
│           risultati_*.xlsx         # File originali con le risposte dei modelli pre-valutazione
│
└── pipeline/                  # Core della valutazione
    └── evaluator_pipeline.py  # Agente Valutatore (Logica di scoring e penalità)

# ⚙️ Flusso di Lavoro (Workflow)

Il processo sperimentale è automatizzato in due stadi principali:

## 1. Fase di Valutazione (Pipeline)

Lo script `pipeline/evaluator_pipeline.py` agisce come un **Agente Valutatore**.

- **Input**: Legge i file Excel grezzi dalla cartella `data/rawdata/`.

- **Processo**:
  - Invia il prompt di valutazione e la risposta dello studente all'LLM Giudice.
  - Effettua il parsing della risposta JSON.
  - Calcola le penalità per allucinazioni (fattore 8x).

- **Audit Trail**: Per ogni riga valutata, salva un file JSON dettagliato in `analysis/output/logs/` per garantire la tracciabilità di ogni decisione.

- **Output**: Genera i file "scored" nella cartella `data/` (es. `risultati_analisi_scored.xlsx`).

## 2. Fase di Analisi (Analysis)

Lo script `analysis/analyze_results_advanced.py` elabora i dati valutati.

- **Input**: Legge i file `*_scored.xlsx` dalla cartella `data/`.

- **Processo**:
  - Esegue test statistici (Kruskal-Wallis, Cohen's d).
  - Rileva outlier statistici (Z-score > 3).
  - Calcola correlazioni (Pearson/Spearman) tra lunghezza risposta, tempo e score.

- **Output**:
  - Aggiorna il report `analysis/analisi_statistica_avanzata.xlsx`.
  - Genera i grafici `.png` nella cartella `analysis/output/figs/`.

---

# 🚀 Guida alla Riproducibilità

Per replicare l'analisi sui dati forniti, seguire questi passaggi:

## Prerequisiti

Installare le librerie Python necessarie:

```bash
pip install pandas numpy scipy matplotlib openpyxl seaborn
```

## Esecuzione Analisi Statistica

Per rigenerare i grafici e le tabelle statistiche presenti nella tesi utilizzando i dati già valutati:

```bash
cd analysis
python analyze_results_advanced.py "../data/*_scored.xlsx"
```

I grafici verranno salvati in `analysis/output/figs/`.

## (Opzionale) Riesecuzione della Valutazione

Se si possiede una API Key valida e si vuole rilanciare la valutazione sui dati grezzi:

```bash
# Imposta la chiave API (es. OpenAI o Perplexity)
export OPENAI_API_KEY="la-tua-chiave"
export EVAL_PROVIDER="openai"  # oppure "perplexity"

# Esegui la pipeline
python pipeline/evaluator_pipeline.py data/rawdata/risultati_analisi.xlsx
```

---

# 📊 Sintesi dei Risultati

L'analisi condotta su 334 osservazioni (consultabile in `executive_summary.md`) evidenzia:

- **Miglior Modello Generalista**: GPT-5 (Score medio: 96.38)
- **Miglior Modello Tecnico (Coding)**: Gemini 2.5 Pro (Score: 100/100 in POO)
- **Tecnica di Prompting**: Il Few-shot risulta la tecnica più stabile ed efficace
- **Efficienza**: Non esiste correlazione significativa tra la lunghezza della risposta (token) e la qualità didattica

---

# 👤 Autore e Riferimenti

**Giuseppe Di Somma**  
Dipartimento di Informatica, Università degli Studi di Salerno  
Anno Accademico 2025-2026

Se utilizzi questo codice o i dati per la tua ricerca, cita la tesi come segue:

```bibtex
@thesis{disomma2026llm,
  author = {Di Somma, Giuseppe},
  title = {Uso dei large language models a supporto dell'insegnamento e della comprensione didattica},
  school = {Università degli Studi di Salerno},
  year = {2026},
  type = {Tesi di Laurea Triennale},
  note = {Available at GitHub: https://github.com/giudsv/LLMEducation}
}
```
