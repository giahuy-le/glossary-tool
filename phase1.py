"""Phase 1: term extraction."""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from .config import NGRAM_MAX
from .text_processing import clean_text, generate_ngrams_from_segment, looks_like_title_variant, tokenize_to_segments


def _detect_text_column(df: pd.DataFrame) -> str | None:
    if "text_en" in df.columns:
        return "text_en"
    for col in df.columns:
        if "text" in col.lower():
            return col
    return None


def run_phase1(input_file: str):
    print("🧩 Running Phase 1: extraction ...")
    df = pd.read_csv(input_file)
    text_col = _detect_text_column(df)
    if not text_col:
        raise RuntimeError("Could not find text column in input CSV.")

    texts = df[text_col].dropna().astype(str).tolist()
    freq = Counter()
    variants = defaultdict(list)
    first_seen = {}
    first_pos = {}

    for row_idx, text in enumerate(texts):
        cleaned = clean_text(text)
        for seg_tokens in tokenize_to_segments(cleaned):
            for ngram_tokens in generate_ngrams_from_segment(seg_tokens, max_n=NGRAM_MAX):
                filtered = [t for t in ngram_tokens if not re_fullmatch_digits(t)]
                if not filtered:
                    continue
                if len(filtered) == 1 and is_single_char(filtered[0]):
                    continue
                key = " ".join([t.lower() for t in filtered])
                freq[key] += 1
                variant = " ".join(filtered)
                if not variants[key] or variants[key][-1] != variant:
                    variants[key].append(variant)
                first_seen.setdefault(key, variant)
                first_pos.setdefault(key, row_idx)

    rows = []
    for key, count in freq.items():
        seen = variants[key]
        disp = next((v for v in seen if looks_like_title_variant(v)), None) or first_seen.get(key, seen[0])
        rows.append({"term": disp, "freq": count, "order": first_pos.get(key, 10**12)})

    df_out = pd.DataFrame(rows).sort_values(by=["order", "term"], ascending=[True, True]).reset_index(drop=True)
    return df_out, texts


def re_fullmatch_digits(token: str) -> bool:
    return bool(token and token.isdigit())


def is_single_char(token: str) -> bool:
    return len(token) == 1 and token.isalnum()

