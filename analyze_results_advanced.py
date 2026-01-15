#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Analysis for LLM Education Experiments (local, no API)

Usage (PowerShell):
  python analyze_results_advanced.py "C:\\path\\to\\*_scored.xlsx"
  (puoi passare più pattern o file)

Requisiti: pandas, numpy. Opzionali: scipy (test non-parametrici), matplotlib (grafici).
Schema Excel atteso: ID, Disciplina, Task, Tecnica, Modello, Prompt, Output_LLM, Score, Rank, Tempo_s, Token, Note_valutatore
"""

import sys, os, glob, math
from typing import List
import numpy as np
import pandas as pd

# Opzionali
try:
    import scipy.stats as sps
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

REQ_COLS = ["ID","Disciplina","Task","Tecnica","Modello","Prompt","Output_LLM","Score","Rank","Tempo_s","Token","Note_valutatore"]

# ------------------- I/O -------------------

def load_concat(patterns: List[str]):
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    if not paths:
        raise SystemExit("Nessun file trovato per i pattern indicati.")
    frames = []
    for p in paths:
        try:
            df = pd.read_excel(p)
            df["__source_file"] = os.path.basename(p)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Impossibile leggere {p}: {e}")
    if not frames:
        raise SystemExit("Nessun file leggibile.")
    return pd.concat(frames, ignore_index=True), paths

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Score","Rank","Tempo_s","Token"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Disciplina","Task","Tecnica","Modello","Prompt","Output_LLM"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df

# ------------------- Sanity checks -------------------

def sanity_checks(df: pd.DataFrame):
    issues = {}
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Mancano colonne richieste: {missing}")

    # Score range/missing
    bad_score = df[(df["Score"] < 0) | (df["Score"] > 100) | (df["Score"].isna())]
    if not bad_score.empty:
        issues["Score_out_of_range_or_missing"] = bad_score

    # Rank anomali per (Disciplina,Task,Tecnica)
    bad_rank_idx = []
    for _, sub in df.groupby(["Disciplina","Task","Tecnica"], dropna=False):
        ranks = pd.to_numeric(sub["Rank"], errors="coerce").dropna()
        if ranks.empty: 
            continue
        n = len(sub)
        # atteso: denso 1..n; inoltre “guard rail” su valori enormi
        if ranks.max() > n or ranks.min() < 1 or ranks.max() > 50:
            bad_rank_idx.extend(sub.index.tolist())
    if bad_rank_idx:
        issues["Rank_anomalies"] = df.loc[sorted(set(bad_rank_idx))]

    # Campi core mancanti
    miss_core = df[df["Score"].isna() | df["Modello"].isna() | df["Tecnica"].isna() | df["Task"].isna()]
    if not miss_core.empty:
        issues["Missing_core_fields"] = miss_core

    return issues

# ------------------- Stat helpers -------------------

def effect_size_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return float("nan")
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    n1, n2 = len(a), len(b)
    s = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / max(1, (n1+n2-2)))
    return 0.0 if s == 0 else (m1 - m2)/s

def effect_size_cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0: return float("nan")
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    n = len(a)*len(b)
    return (gt - lt)/n if n else float("nan")

def stats_by(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    agg = (df.groupby(by, dropna=False)
             .agg(MeanScore=("Score","mean"),
                  StdScore=("Score","std"),
                  N=("Score","count"),
                  WinRate=("Rank", lambda s: (pd.to_numeric(s, errors="coerce")==1).mean() if len(s)>0 else np.nan),
                  MeanRank=("Rank","mean"),
                  MeanTime=("Tempo_s","mean"),
                  MeanToken=("Token","mean"))
             .reset_index())
    return agg.sort_values(["MeanScore","WinRate"], ascending=[False,False])

# ------------------- Pattern mining -------------------

def technique_analysis_within_model(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, subm in df.groupby("Modello"):
        tech_groups = {t: pd.to_numeric(g["Score"], errors="coerce").dropna().values
                       for t, g in subm.groupby("Tecnica")}
        techniques = sorted(tech_groups.keys())
        # Test complessivo
        if len(techniques) >= 3 and HAS_SCIPY:
            samples = [tech_groups[t] for t in techniques if len(tech_groups[t])>1]
            if len(samples) >= 3:
                try:
                    H, p = sps.kruskal(*samples)  # robusto
                    overall = ("kruskal", float(H), float(p))
                except Exception:
                    overall = ("kruskal", float("nan"), float("nan"))
            else:
                overall = ("kruskal", float("nan"), float("nan"))
        else:
            overall = ("none", float("nan"), float("nan"))

        # Pairwise
        for i, t1 in enumerate(techniques):
            for t2 in techniques[i+1:]:
                a = tech_groups[t1]; b = tech_groups[t2]
                if len(a)==0 or len(b)==0:
                    d = np.nan; delta = np.nan; pval = np.nan
                else:
                    d  = effect_size_cohens_d(a, b)
                    delta = effect_size_cliffs_delta(a, b)
                    if HAS_SCIPY:
                        try:
                            _, pval = sps.mannwhitneyu(a, b, alternative="two-sided")
                            pval = float(pval)
                        except Exception:
                            pval = np.nan
                    else:
                        pval = np.nan
                rows.append({
                    "Modello": model, "Pair": f"{t1} vs {t2}",
                    "Cohens_d": float(d) if not np.isnan(d) else None,
                    "Cliffs_delta": float(delta) if not np.isnan(delta) else None,
                    "MannWhitney_p": pval,
                    "OverallTest": overall[0], "OverallStat": overall[1], "OverallP": overall[2],
                    "Mean_t1": float(np.nanmean(a)) if len(a) else None,
                    "Mean_t2": float(np.nanmean(b)) if len(b) else None,
                    "N_t1": int(len(a)), "N_t2": int(len(b))
                })
    return pd.DataFrame(rows)

def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Prompt_len"] = out["Prompt"].apply(lambda s: len(str(s)) if not pd.isna(s) else 0)
    out["Output_len"] = out["Output_LLM"].apply(lambda s: len(str(s)) if not pd.isna(s) else 0)
    return out

def correlations(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    metrics = ["Tempo_s","Token","Prompt_len","Output_len"]
    rows = []
    for key, sub in df.groupby(by, dropna=False):
        sub = sub.copy()
        sub["Score"] = pd.to_numeric(sub["Score"], errors="coerce")
        if sub["Score"].notna().sum() < 3:
            continue

        for m in metrics:
            x = pd.to_numeric(sub[m], errors="coerce")
            y = sub["Score"]
            valid = x.notna() & y.notna()
            x = x[valid].values
            y = y[valid].values
            if len(x) < 3:
                continue

            # se la varianza è zero, ogni correlazione è indefinita
            if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                pr = np.nan; spr = np.nan; sp_p = np.nan; slope = np.nan
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    # Pearson
                    try:
                        pr = float(np.corrcoef(x, y)[0, 1])
                    except Exception:
                        pr = np.nan
                    # Spearman
                    if HAS_SCIPY:
                        try:
                            spr, sp_p = sps.spearmanr(x, y)
                            spr = float(spr); sp_p = float(sp_p)
                        except Exception:
                            spr = np.nan; sp_p = np.nan
                    else:
                        spr = np.nan; sp_p = np.nan
                    # Regressione lineare (slope)
                    try:
                        slope, _ = np.polyfit(x, y, 1)
                        slope = float(slope)
                    except Exception:
                        slope = np.nan

            row = {}
            if isinstance(key, tuple):
                for i, kname in enumerate(by):
                    row[kname] = key[i]
            else:
                row[by[0]] = key
            row.update({
                "Metric": m,
                "N": int(len(x)),
                "Pearson_r": pr,
                "Spearman_r": spr,
                "Spearman_p": sp_p,
                "Slope": slope
            })
            rows.append(row)
    return pd.DataFrame(rows)


def task_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(["Disciplina","Task","Tecnica"], dropna=False):
        sub = sub.copy()
        sub["__Score"] = pd.to_numeric(sub["Score"], errors="coerce")
        sub["__Rank"]  = pd.to_numeric(sub["Rank"], errors="coerce")
        sub = sub.dropna(subset=["__Score"])
        if sub.empty: continue
        sub = sub.sort_values(by=["__Score","__Rank"], ascending=[False,True])
        top = sub.iloc[0]
        rows.append({
            "Disciplina": key[0], "Task": key[1], "Tecnica": key[2],
            "BestModel": top.get("Modello",""), "BestScore": float(top["__Score"]),
            "BestRank": float(top["__Rank"]) if not math.isnan(top["__Rank"]) else None,
            "EsempioFile": top.get("__source_file","")
        })
    return pd.DataFrame(rows).sort_values(["Disciplina","Task","Tecnica"]).reset_index(drop=True)

def outliers_by_group(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(by, dropna=False):
        sc = pd.to_numeric(sub["Score"], errors="coerce")
        mu = sc.mean(); sd = sc.std(ddof=1)
        if np.isnan(sd) or sd == 0: continue
        z = (sc - mu)/sd
        idx = sub.index[(np.abs(z) >= 3.0)].tolist()
        for i in idx:
            row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
            row.update({
                "ID": sub.loc[i, "ID"],
                "Modello": sub.loc[i, "Modello"],
                "Score": float(sub.loc[i, "Score"]),
                "z_score": float((sub.loc[i, "Score"] - mu)/sd),
                "__source_file": sub.loc[i, "__source_file"]
            })
            rows.append(row)
    return pd.DataFrame(rows)

# ------------------- Export -------------------

def save_excel(df_all: pd.DataFrame, out_xlsx: str):
    df_feat = add_length_features(df_all)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xl:
        # Overview
        overview = pd.DataFrame({
            "Metric": ["Files", "Total Rows", "Disciplines", "Tasks", "Techniques", "Models"],
            "Value": [
                df_feat["__source_file"].nunique(),
                len(df_feat),
                df_feat["Disciplina"].nunique(),
                df_feat["Task"].nunique(),
                df_feat["Tecnica"].nunique(),
                df_feat["Modello"].nunique(),
            ]
        })
        overview.to_excel(xl, sheet_name="Overview", index=False)

        # Sanity
        issues = sanity_checks(df_feat)
        if issues:
            for name, sub in issues.items():
                sub.to_excel(xl, sheet_name=f"Issue_{name[:25]}", index=False)
        else:
            pd.DataFrame({"Info":["Nessuna anomalia rilevante"]}).to_excel(xl, sheet_name="Issues", index=False)

        # Aggregazioni principali
        stats_by(df_feat, ["Modello"]).to_excel(xl, sheet_name="Model_Overall", index=False)
        stats_by(df_feat, ["Disciplina","Modello"]).to_excel(xl, sheet_name="Model_by_Discipline", index=False)
        stats_by(df_feat, ["Tecnica"]).to_excel(xl, sheet_name="Technique_Overall", index=False)
        stats_by(df_feat, ["Modello","Tecnica"]).to_excel(xl, sheet_name="Model_by_Technique", index=False)
        stats_by(df_feat, ["Disciplina","Tecnica"]).to_excel(xl, sheet_name="Technique_by_Discipline", index=False)
        stats_by(df_feat, ["Disciplina","Task","Modello"]).to_excel(xl, sheet_name="Model_by_Task", index=False)
        stats_by(df_feat, ["Disciplina","Task","Tecnica"]).to_excel(xl, sheet_name="Technique_by_Task", index=False)

        # Leaderboard per Task
        task_leaderboard(df_feat).to_excel(xl, sheet_name="Task_Leaderboard", index=False)

        # Effetti tecniche dentro i modelli + Correlazioni
        te = technique_analysis_within_model(df_feat)
        if not te.empty:
            te.to_excel(xl, sheet_name="Technique_Effects", index=False)
        correlations(df_feat, ["Modello"]).to_excel(xl, sheet_name="Corr_by_Model", index=False)
        correlations(df_feat, ["Modello","Disciplina"]).to_excel(xl, sheet_name="Corr_by_Model_Disc", index=False)

        # Outlier
        out_mdl = outliers_by_group(df_feat, ["Modello"])
        if not out_mdl.empty:
            out_mdl.to_excel(xl, sheet_name="Outliers_by_Model", index=False)
        out_task = outliers_by_group(df_feat, ["Disciplina","Task","Tecnica"])
        if not out_task.empty:
            out_task.to_excel(xl, sheet_name="Outliers_by_Task", index=False)

        # Raw
        df_feat.to_excel(xl, sheet_name="Raw_With_Features", index=False)

def make_charts(df_all: pd.DataFrame, out_dir="figs"):
    if not HAS_MPL:
        print("[WARN] matplotlib non disponibile: salto grafici."); return
    import numpy as np
    import os
    os.makedirs(out_dir, exist_ok=True)

    def _bar(series, title, ylabel, fname):
        plt.figure()
        series.plot(kind="bar")
        plt.title(title); plt.ylabel(ylabel); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname)); plt.close()

    # 1) Score medio per modello
    if {"Modello","Score"}.issubset(df_all.columns):
        m = df_all.groupby("Modello")["Score"].mean().sort_values(ascending=False)
        _bar(m, "Score medio per modello", "Score medio", "score_medio_per_modello.png")

    # 2) Win-rate per modello (Rank=1)
    if {"Modello","Rank"}.issubset(df_all.columns):
        wr = df_all.assign(_r=pd.to_numeric(df_all["Rank"], errors="coerce")) \
                   .groupby("Modello")["_r"].apply(lambda s: (s==1).mean()) \
                   .sort_values(ascending=False)
        _bar(wr, "Win-rate (Rank=1) per modello", "Frazione di vittorie", "winrate_per_modello.png")

    # 3) Score medio per tecnica
    if {"Tecnica","Score"}.issubset(df_all.columns):
        t = df_all.groupby("Tecnica")["Score"].mean().sort_values(ascending=False)
        _bar(t, "Score medio per tecnica", "Score medio", "score_medio_per_tecnica.png")

    # 4) Boxplot degli score per modello
    if {"Modello","Score"}.issubset(df_all.columns):
        plt.figure()
        order = df_all.groupby("Modello")["Score"].mean().sort_values(ascending=False).index.tolist()
        data = [pd.to_numeric(df_all.loc[df_all["Modello"]==m, "Score"], errors="coerce").dropna().values for m in order]
        plt.boxplot(data, labels=order, vert=True, showfliers=True)
        plt.title("Distribuzione degli Score per modello"); plt.ylabel("Score"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "boxplot_score_per_modello.png")); plt.close()

    # 5) Boxplot degli score per tecnica
    if {"Tecnica","Score"}.issubset(df_all.columns):
        plt.figure()
        order = df_all.groupby("Tecnica")["Score"].mean().sort_values(ascending=False).index.tolist()
        data = [pd.to_numeric(df_all.loc[df_all["Tecnica"]==tec, "Score"], errors="coerce").dropna().values for tec in order]
        plt.boxplot(data, labels=order, vert=True, showfliers=True)
        plt.title("Distribuzione degli Score per tecnica"); plt.ylabel("Score"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "boxplot_score_per_tecnica.png")); plt.close()

    # 6) Heatmap Disciplina × Modello (media score) con imshow
    if {"Disciplina","Modello","Score"}.issubset(df_all.columns):
        pivot = df_all.pivot_table(index="Disciplina", columns="Modello", values="Score", aggfunc="mean")
        if pivot.shape[0] and pivot.shape[1]:
            plt.figure()
            plt.imshow(pivot, aspect="auto")
            plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
            plt.yticks(range(pivot.shape[0]), pivot.index)
            plt.title("Score medio per Disciplina × Modello")
            plt.colorbar(label="Score medio")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "heatmap_disciplina_modello.png")); plt.close()

    # 7) Scatter Token vs Score con regressione
    if {"Token","Score"}.issubset(df_all.columns):
        valid = df_all[["Token","Score"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(valid) >= 3:
            x, y = valid["Token"].values, valid["Score"].values
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100); ys = slope*xs + intercept
            plt.figure()
            plt.scatter(x, y, s=10); plt.plot(xs, ys)
            plt.title("Relazione tra Token e Score (con regressione)")
            plt.xlabel("Token"); plt.ylabel("Score"); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "scatter_token_score_regressione.png")); plt.close()

    # 8) Scatter Tempo_s vs Score con regressione
    if {"Tempo_s","Score"}.issubset(df_all.columns):
        valid = df_all[["Tempo_s","Score"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(valid) >= 3:
            x, y = valid["Tempo_s"].values, valid["Score"].values
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100); ys = slope*xs + intercept
            plt.figure()
            plt.scatter(x, y, s=10); plt.plot(xs, ys)
            plt.title("Relazione tra Tempo_s e Score (con regressione)")
            plt.xlabel("Tempo_s"); plt.ylabel("Score"); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "scatter_tempo_score_regressione.png")); plt.close()


def write_summary_md(df_all: pd.DataFrame, out_md="executive_summary.md"):
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Executive Summary – Analisi Risultati LLM\n\n")
        f.write(f"- Files: {df_all['__source_file'].nunique()}  \n")
        f.write(f"- Righe totali: {len(df_all)}  \n")
        f.write(f"- Modelli: {df_all['Modello'].nunique()} | Discipline: {df_all['Disciplina'].nunique()} | Tecniche: {df_all['Tecnica'].nunique()} | Task: {df_all['Task'].nunique()}\n\n")
        g = df_all.groupby("Modello")["Score"].mean().sort_values(ascending=False)
        if len(g) > 0:
            f.write("## Modello complessivamente migliore (Score medio)\n\n")
            for mdl, val in g.items():
                f.write(f"- {mdl}: {val:.2f}\n")
            f.write("\n")
        f.write("## Note su pattern possibili\n")
        f.write("- Confronta le tecniche dentro ciascun modello nel foglio 'Technique_Effects'.\n")
        f.write("- Verifica correlazioni tra Score e Tempo/Token/Prompt_len/Output_len nei fogli 'Corr_by_Model*'.\n")
        f.write("- Controlla gli outlier nei fogli 'Outliers_*' per comprendere variabilità anomala.\n")

# ------------------- Main -------------------

def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Uso: python analyze_results_advanced.py \"C:\\\\path\\\\to\\\\*_scored.xlsx\"")
        return 1
    df_all, _ = load_concat(argv[1:])
    for c in REQ_COLS:
        if c not in df_all.columns:
            raise SystemExit(f"Manca la colonna richiesta: {c}")
    df_all = coerce_types(df_all)

    out_xlsx = "analisi_statistica_avanzata.xlsx"
    save_excel(df_all, out_xlsx)
    print(f"[OK] Report Excel: {out_xlsx}")

    make_charts(df_all, out_dir="figs")
    write_summary_md(df_all, out_md="executive_summary.md")
    print("[OK] Executive summary: executive_summary.md")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
