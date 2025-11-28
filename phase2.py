"""Phase 2: normalization, deduplication and filtering."""

from __future__ import annotations

import re

import inflect
import pandas as pd

from .config import CAPITAL_PRESENCE_REQUIRED, MIN_FREQ
from .text_processing import (
    build_context_string,
    build_phase1_segment_set,
    has_phase1_clean_segment,
    starts_with_capital_first_token,
)

_inflector = inflect.engine()


def normalize_key(term: str) -> str:
    if not isinstance(term, str) or not term.strip():
        return ""
    s = re_sub_possessive(term.strip())
    s = re_sub_separators(s)
    s = re_sub_non_word(s)
    s = re_sub_spaces(s)
    toks = []
    for w in s.split():
        sg = _inflector.singular_noun(w)
        toks.append(sg if sg else w)
    return " ".join(toks).lower()


def run_phase2(df_out: pd.DataFrame, texts: list[str]):
    print("🧩 Running Phase 2: normalization, deduplication & filtering ...")
    df = df_out.copy()
    if "order" not in df.columns:
        df["order"] = range(len(df))

    df["_norm"] = df["term"].astype(str).map(normalize_key)
    df["_len"] = df["term"].astype(str).map(len)
    df_sorted = df.sort_values(by=["_norm", "_len", "order"], ascending=[True, True, True])
    df_norm = df_sorted.groupby("_norm", as_index=False).first()
    df_norm = df_norm.drop(columns=["_norm", "_len"]).sort_values(by=["order", "term"]).reset_index(drop=True)

    df_norm = df_norm[df_norm["freq"] >= MIN_FREQ]
    if CAPITAL_PRESENCE_REQUIRED:
        df_norm = df_norm[df_norm["term"].astype(str).str.contains(r"[A-Z]")]
    df_norm = df_norm[df_norm["term"].astype(str).map(starts_with_capital_first_token)]

    segset = build_phase1_segment_set(texts)

    must_keep_vals, context_vals = [], []
    for _, row in df_norm.iterrows():
        t = str(row["term"])
        f = int(row["freq"])
        if f < MIN_FREQ:
            must_keep_vals.append(False)
            context_vals.append("")
            continue
        mk = has_phase1_clean_segment(t, segset)
        ctx = build_context_string(t, texts, max_lines=30, char_cap=1200)
        must_keep_vals.append(bool(mk))
        context_vals.append(ctx)

    df_norm["must_keep"] = must_keep_vals
    df_norm["context"] = context_vals

    protect = set(df_norm.loc[df_norm["must_keep"] == True, "term"].astype(str))
    df_norm, _ = prune_parent_child_terms(df_norm, protect=protect)
    return df_norm


def prune_parent_child_terms(df: pd.DataFrame, protect: set[str] | None = None):
    protect = protect or set()
    terms = df["term"].tolist()
    freq_map = dict(zip(df["term"], df["freq"]))
    removed = set()

    for t in terms:
        if (t.endswith("'s") or t.endswith("’s")) and (t not in protect):
            removed.add(t)

    sorted_terms = sorted(terms, key=lambda t: len(t.split()), reverse=True)

    for i, t_long in enumerate(sorted_terms):
        if t_long in removed:
            continue
        tokens_long = t_long.split()
        freq_long = freq_map.get(t_long, 0)

        for t_short in sorted_terms[i + 1:]:
            if t_short in removed or t_short == t_long:
                continue
            if t_short in protect:
                continue

            tokens_short = t_short.split()
            freq_short = freq_map.get(t_short, 0)

            for j in range(len(tokens_long) - len(tokens_short) + 1):
                if tokens_long[j:j + len(tokens_short)] == tokens_short:
                    if freq_short <= freq_long:
                        removed.add(t_short)
                    break

    kept = [t for t in terms if t not in removed]
    return df[df["term"].isin(kept)].reset_index(drop=True), removed


def re_sub_possessive(s: str) -> str:
    return re.sub(r"[’']s\b", "", s)


def re_sub_separators(s: str) -> str:
    return re.sub(r"[_\-\/]+", " ", s)


def re_sub_non_word(s: str) -> str:
    return re.sub(r"[^\w\s']", "", s)


def re_sub_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

