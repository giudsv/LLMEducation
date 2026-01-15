# LLMEducation
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
