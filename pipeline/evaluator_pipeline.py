#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluator Pipeline - Perplexity/OpenAI Ready (v1.1)

- Provider via env: EVAL_PROVIDER=perplexity|openai (default: perplexity)
- Compatibile con OpenAI Chat Completions
- Parser robusto: rimuove <think>...</think> ed estrae il primo JSON valido { ... }
- Salva i JSON grezzi in ./logs/<file>_row<ID>.json (audit)
- Compila Score, Note_valutatore, Tempo_s, Token e calcola Rank (1–3) per (Disciplina, Task, Modello)

Excel richiesto: ID, Disciplina, Task, Tecnica, Modello, Output_LLM, Score, Rank, Tempo_s, Token, Note_valutatore, Prompt
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd

# =========================
# Config
# =========================

EVAL_PROVIDER = os.getenv("EVAL_PROVIDER", "perplexity").strip().lower()
EVAL_MODEL = os.getenv("EVAL_MODEL", "sonar-pro" if EVAL_PROVIDER == "perplexity" else "gpt-5-thinking")
MAX_RETRY = int(os.getenv("EVAL_MAX_RETRY", "3"))
RETRY_SLEEP = float(os.getenv("EVAL_RETRY_SLEEP", "2.0"))
USE_API = os.getenv("USE_OPENAI", "1") == "1"
API_KEY = os.getenv("OPENAI_API_KEY", "")  # usa la key Perplexity quando provider=perplexity
DRY_RUN = os.getenv("EVAL_DRY_RUN", "0") == "1"
EVAL_TIMEOUT = float(os.getenv("EVAL_TIMEOUT", "60"))      # sec, timeout per request
EVAL_PACING  = float(os.getenv("EVAL_PACING", "0.0"))      # sec, sleep tra chiamate
EVAL_SAVE_EVERY = int(os.getenv("EVAL_SAVE_EVERY", "25"))  # salva ogni N righe
EVAL_MAX_ROWS = int(os.getenv("EVAL_MAX_ROWS", "0"))       # 0 = no limite, >0 elabora solo prime N righe


# Endpoint
PPLX_BASE_URL   = os.getenv("PPLX_BASE_URL", "https://api.perplexity.ai")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

REQUIRED_COLS = [
    "ID", "Disciplina", "Task", "Tecnica", "Modello",
    "Output_LLM", "Score", "Rank", "Tempo_s", "Token", "Note_valutatore", "Prompt"
]

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)

# =========================
# Prompts e rubriche
# =========================

SYSTEM_PROMPT = (
    "Sei un valutatore rigoroso e imparziale. Valuti risposte di LLM per task educativi. "
    "Applichi SOLO la rubrica fornita, penalizzi allucinazioni/imprecisioni/violazioni di formato. "
    "Produci esclusivamente un JSON valido secondo lo schema dato, senza testo extra. "
    "Mantieni coerenza, cita errori in \"note_valutatore\" in modo operativo."
)

EVAL_SCHEMA_TEXT = """
Schema JSON obbligatorio (nessun testo fuori dal JSON):
{
  "task": "T1|T2|...|T7",
  "disciplina": "Analisi|Reti di Calcolatori|POO|IA-LLM",
  "modello_risposto": "GPT-5|Claude|Gemini|Sonar",
  "tecnica": "Zero-shot|Few-shot|CoT",
  "criteri": {
    "correttezza": 0-5,
    "completezza": 0-5,
    "chiarezza": 0-5,
    "aderenza_istruzioni": 0-5,
    "specificita_disciplinare": 0-5
  },
  "penalita": {
    "allucinazioni": 0-3,
    "imprecisioni": 0-2,
    "violazioni_format": 0-1
  },
  "peso_criteri": {
    "correttezza": 0.35,
    "completezza": 0.20,
    "chiarezza": 0.15,
    "aderenza_istruzioni": 0.15,
    "specificita_disciplinare": 0.15
  },
  "score": {
    "parziale": 0-5,
    "penalty": 0-6,
    "totale_100": 0-100
  },
  "note_valutatore": "string",
  "tempo_s": 0,
  "token": 0
}
Calcolo da applicare:
- parziale = somma(criterio_i * peso_i) su scala 0-5
- penalty = allucinazioni + imprecisioni + violazioni_format  (0-6)
- totale_100 = max(0, round( (parziale/5)*100 - penalty*8 ))
"""

def student_eval_user(disciplina: str, task: str, modello: str, tecnica: str,
                      prompt_input: str, output_llm: str) -> str:
    return (
        "Contesto:\n"
        f"- Disciplina: {disciplina}\n"
        f"- Task: {task}\n"
        f"- Modello che ha risposto: {modello}\n"
        f"- Tecnica: {tecnica}\n"
        f"- Istruzioni date al modello (prompt input): {prompt_input}\n"
        f"- Risposta del modello da valutare: {output_llm}\n\n"
        "Rubrica e pesi:\n"
        "correttezza(0-5; 0.35), completezza(0-5; 0.20), chiarezza(0-5; 0.15), "
        "aderenza_istruzioni(0-5; 0.15), specificita_disciplinare(0-5; 0.15).\n"
        "Penalita: allucinazioni(0-3), imprecisioni(0-2), violazioni_format(0-1).\n"
        "Calcola parziale, penalty, totale_100 come da definizione.\n\n"
        "Restituisci SOLO il JSON nello schema richiesto.\n"
        f"{EVAL_SCHEMA_TEXT}"
    )

def teacher_eval_user(disciplina: str, task: str, modello: str, tecnica: str,
                      prompt_input: str, output_llm: str) -> str:
    return (
        "Contesto:\n"
        f"- Disciplina: {disciplina}\n"
        f"- Task: {task}\n"
        f"- Modello che ha risposto: {modello}\n"
        f"- Tecnica: {tecnica}\n"
        f"- Istruzioni date al modello (prompt input): {prompt_input}\n"
        f"- Risposta del modello da valutare: {output_llm}\n\n"
        "Rubrica e pesi come nello schema base. Penalita come definite.\n"
        "Calcola parziale, penalty, totale_100.\n\n"
        "Restituisci SOLO il JSON nello schema richiesto.\n"
        f"{EVAL_SCHEMA_TEXT}"
    )

def pick_agent(task: str) -> str:
    if task in ("T1", "T2", "T3", "T4"):
        return "student"
    elif task in ("T5", "T6", "T7"):
        return "teacher"
    return "student"

# =========================
# Client OpenAI-compatible
# =========================

def get_client():
    if not USE_API or not API_KEY:
        return None, None
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("SDK openai non disponibile. Installa con: pip install openai>=1.0.0") from e

    if EVAL_PROVIDER == "perplexity":
        client = OpenAI(api_key=API_KEY, base_url=PPLX_BASE_URL)
        base = "perplexity"
    else:
        client = OpenAI(api_key=API_KEY, base_url=OPENAI_BASE_URL)
        base = "openai"
    return client, base

# =========================
# Helpers
# =========================

def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

def extract_first_json(text: str) -> str:
    text = text.strip()
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : i + 1]
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1 and b > a:
        return text[a : b + 1]
    raise ValueError("Nessun JSON trovato nel testo di risposta.")

def save_raw_json(row_id: Any, filename_stem: str, content: str) -> None:
    out = LOG_DIR / f"{filename_stem}_row{row_id}.json"
    out.write_text(content, encoding="utf-8")

def validate_eval_json(d: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        for k in ("task", "disciplina", "modello_risposto", "tecnica", "criteri", "penalita", "peso_criteri", "score", "note_valutatore"):
            if k not in d: return False, f"Chiave mancante: {k}"
        for k in ("correttezza", "completezza", "chiarezza", "aderenza_istruzioni", "specificita_disciplinare"):
            if k not in d["criteri"]: return False, f"Criterio mancante: {k}"
        for k in ("allucinazioni", "imprecisioni", "violazioni_format"):
            if k not in d["penalita"]: return False, f"Penalita mancante: {k}"
        for k in ("parziale", "penalty", "totale_100"):
            if k not in d["score"]: return False, f"Campo score mancante: {k}"
        tot = float(d["score"]["totale_100"])
        if not (0.0 <= tot <= 100.0):
            return False, "Score totale_100 fuori range 0..100"
    except Exception as e:
        return False, f"Errore validazione: {e}"
    return True, "ok"

def compute_ranks(group_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    sortable = []
    for r in group_rows:
        score = r.get("Score")
        try:
            val = float(score)
        except Exception:
            val = -1.0
        sortable.append((val, r))
    sortable.sort(key=lambda t: t[0], reverse=True)

    rank_map = {}
    rank = 1
    for _, row in sortable:
        tech = row.get("Tecnica")
        if tech is not None:
            rank_map[tech] = rank
            rank += 1
    return rank_map

# =========================
# LLM call
# =========================

def llm_call(messages: list, model: str, filename_stem: str = "file", row_id: any = 0):
    # Dry-run
    if DRY_RUN or not USE_API or not API_KEY:
        mock = {
            "task": "T1",
            "disciplina": "Mock",
            "modello_risposto": "Mock",
            "tecnica": "Zero-shot",
            "criteri": {"correttezza": 4.0, "completezza": 3.5, "chiarezza": 4.0, "aderenza_istruzioni": 4.0, "specificita_disciplinare": 3.5},
            "penalita": {"allucinazioni": 0, "imprecisioni": 0, "violazioni_format": 0},
            "peso_criteri": {"correttezza": 0.35, "completezza": 0.20, "chiarezza": 0.15, "aderenza_istruzioni": 0.15, "specificita_disciplinare": 0.15},
            "score": {"parziale": 3.8, "penalty": 0, "totale_100": 77},
            "note_valutatore": "Mock: pipeline in dry-run.",
            "tempo_s": 0.0,
            "token": 0
        }
        content = json.dumps(mock, ensure_ascii=False)
        save_raw_json(row_id, filename_stem, content)
        return content, 0.0, 0

    from openai import OpenAI
    base_url = PPLX_BASE_URL if EVAL_PROVIDER == "perplexity" else OPENAI_BASE_URL
    client  = OpenAI(api_key=API_KEY, base_url=base_url)

    tokens_total = 0
    elapsed_s = 0.0

    for attempt in range(1, MAX_RETRY + 1):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=800
                # niente response_format con Perplexity
            )
            elapsed_s = time.time() - t0

            # contenuto
            content = resp.choices[0].message.content or ""
            content = strip_think_blocks(content)
            save_raw_json(row_id, filename_stem, content)

            # token se disponibili
            try:
                # OpenAI SDK >=1 espone resp.usage.{prompt_tokens,completion_tokens,total_tokens}
                if getattr(resp, "usage", None) and getattr(resp.usage, "total_tokens", None) is not None:
                    tokens_total = int(resp.usage.total_tokens)
                elif isinstance(resp, dict) and "usage" in resp and "total_tokens" in resp["usage"]:
                    tokens_total = int(resp["usage"]["total_tokens"])
            except Exception:
                tokens_total = 0

            return content, float(elapsed_s), int(tokens_total)

        except Exception:
            if attempt >= MAX_RETRY:
                raise
            time.sleep(RETRY_SLEEP * attempt)



# =========================
# Pipeline
# =========================

def evaluate_file(path: str, out_suffix: str = "_scored") -> str:
    import pandas as pd
    from pathlib import Path
    import json
    import time

    print(f"[INFO] Loading: {path}")
    df = pd.read_excel(path)

    # Colonne richieste
    REQUIRED_COLS = [
        "ID", "Disciplina", "Task", "Tecnica", "Modello",
        "Output_LLM", "Score", "Rank", "Tempo_s", "Token", "Note_valutatore", "Prompt"
    ]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    # Forza tipi (evita FutureWarning)
    safe_str_cols = ["Disciplina", "Task", "Tecnica", "Modello", "Output_LLM", "Note_valutatore", "Prompt"]
    for col in safe_str_cols:
        if col in df.columns:
            df[col] = df[col].astype("object")
    safe_num_cols = ["Score", "Rank", "Tempo_s", "Token"]
    for col in safe_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df2 = df.copy()
    filename_stem = Path(path).stem

    total_rows = len(df2)
    processed = 0
    started = time.time()

    def _recalc_rank_inplace(dfx: pd.DataFrame):
        dfx["__Score_num"] = pd.to_numeric(dfx["Score"], errors="coerce")
        dfx["Rank"] = (
            dfx.groupby(["Disciplina", "Task", "Tecnica"])["__Score_num"]
               .rank(method="dense", ascending=False)
               .astype("Int64")
        )
        dfx.drop(columns=["__Score_num"], inplace=True)

    def _save_partial(dfx: pd.DataFrame, tag: str = "_partial"):
        tmp_path = str(Path(path).with_name(Path(path).stem + tag + Path(path).suffix))
        dfx.to_excel(tmp_path, index=False)
        print(f"[SAVE] Parziale: {tmp_path}")

    for idx, row in df2.iterrows():
        # Limite righe: se impostato e raggiunto → stop
        if EVAL_MAX_ROWS > 0 and processed >= EVAL_MAX_ROWS:
            print(f"[INFO] Raggiunto limite EVAL_MAX_ROWS={EVAL_MAX_ROWS}.")
            break

        disciplina   = str(row["Disciplina"]).strip()
        task         = str(row["Task"]).strip()
        tecnica      = str(row["Tecnica"]).strip()
        modello      = str(row["Modello"]).strip()
        output_llm   = "" if pd.isna(row["Output_LLM"]) else str(row["Output_LLM"])
        prompt_input = "" if pd.isna(row["Prompt"]) else str(row["Prompt"])
        row_id       = row.get("ID", idx + 1)

        print(f"[{processed+1}/{total_rows}] ID={row_id} {disciplina} {task} {tecnica} {modello}")

        # Nessun output da valutare → skip
        if not output_llm:
            print("  - Nessun Output_LLM: skip")
            processed += 1
            # pacing anche sugli skip per non martellare
            if EVAL_PACING > 0:
                time.sleep(EVAL_PACING)
            # salvataggio parziale
            if EVAL_SAVE_EVERY > 0 and processed % EVAL_SAVE_EVERY == 0:
                _recalc_rank_inplace(df2)
                _save_partial(df2)
            continue

        # Già valutato (score valido) → skip
        pre_score = row.get("Score", None)
        try:
            pre_val = float(pre_score)
        except Exception:
            pre_val = None
        if pre_val is not None and 0 <= pre_val <= 100:
            print("  - Già valutato: skip")
            processed += 1
            if EVAL_PACING > 0:
                time.sleep(EVAL_PACING)
            if EVAL_SAVE_EVERY > 0 and processed % EVAL_SAVE_EVERY == 0:
                _recalc_rank_inplace(df2)
                _save_partial(df2)
            continue

        # Messaggi per l’agente valutatore
        agent = pick_agent(task)  # "student" o "teacher"
        if agent == "student":
            user_prompt = student_eval_user(disciplina, task, modello, tecnica, prompt_input, output_llm)
        else:
            user_prompt = teacher_eval_user(disciplina, task, modello, tecnica, prompt_input, output_llm)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Pacing anti rate-limit
        if EVAL_PACING > 0:
            time.sleep(EVAL_PACING)

        # Chiamata LLM (può restituire content o (content, elapsed, tokens))
        call_result = llm_call(messages, EVAL_MODEL, filename_stem, row_id)
        if isinstance(call_result, tuple) and len(call_result) >= 3:
            content, elapsed_s, tokens_total = call_result[0], float(call_result[1]), int(call_result[2])
        else:
            content, elapsed_s, tokens_total = call_result, 0.0, 0

        # Parsing JSON
        try:
            json_text = extract_first_json(content)
            data = json.loads(json_text)
        except Exception:
            data = json.loads(content)

        # Validazione JSON
        ok, msg = validate_eval_json(data)
        if not ok:
            raise ValueError(f"Invalid evaluation JSON at row {idx+2}: {msg}\n{json.dumps(data, ensure_ascii=False)}")

        # Scrittura valori
        df2.at[idx, "Score"] = float(data["score"]["totale_100"])
        df2.at[idx, "Note_valutatore"] = str(data.get("note_valutatore", "") or "")

        # tempo/token: preferisci quelli dell’agente se >0, altrimenti misurati
        tempo_from_agent = data.get("tempo_s", 0) or 0
        token_from_agent = data.get("token", 0) or 0
        tempo_final = float(tempo_from_agent) if float(tempo_from_agent) > 0 else float(elapsed_s)
        token_final = int(token_from_agent) if int(token_from_agent) > 0 else int(tokens_total)
        df2.at[idx, "Tempo_s"] = tempo_final
        df2.at[idx, "Token"] = token_final

        processed += 1

        # Salvataggi parziali ogni N righe (con rank ricalcolato)
        if EVAL_SAVE_EVERY > 0 and processed % EVAL_SAVE_EVERY == 0:
            _recalc_rank_inplace(df2)
            _save_partial(df2)

    # ======= RICALCOLO RANK FINALE =======
    _recalc_rank_inplace(df2)
    # =====================================

    out_path = str(Path(path).with_name(Path(path).stem + out_suffix + Path(path).suffix))
    df2.to_excel(out_path, index=False)
    elapsed_all = time.time() - started
    print(f"[OK] Saved: {out_path}  (in {elapsed_all:.1f}s)")
    return out_path



def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python evaluator_pipeline_perplexity.py file1.xlsx [file2.xlsx ...]")
        print("Provider: EVAL_PROVIDER=perplexity|openai (default: perplexity)")
        print("Model:    EVAL_MODEL=sonar-pro|sonar|sonar-reasoning|sonar-reasoning-pro ...")
        return 1

    for p in argv[1:]:
        try:
            evaluate_file(p)
        except Exception as e:
            print(f"[ERROR] {p}: {e}")
            return 2
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
