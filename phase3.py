"""Phase 3: AI assisted refinement."""

from __future__ import annotations

import pandas as pd

from .ai import ai_classify_with_context, ai_prune_redundant_terms
from .config import OUTPUT_PHASE3
from .text_processing import split_segments_strict


def run_phase3(df_norm: pd.DataFrame, texts):
    print("🧩 Running Phase 3: AI classify + prune ...")

    df_locked = df_norm[df_norm.get("must_keep", False) == True].copy()
    df_candidates = df_norm[df_norm.get("must_keep", False) == False].copy()

    if df_candidates.empty:
        df_final = df_locked.copy()
        df_final = df_final[["term", "freq", "context"]]
        df_final.to_csv(OUTPUT_PHASE3, index=False, encoding="utf-8-sig")
        print(f"✅ Phase 3 skipped (all must_keep). Wrote {len(df_final)} rows.")
        return df_final

    all_terms = list(df_candidates["term"].astype(str))
    contexts_cache = {
        row.term: split_segments_strict(row.context)[:30] if isinstance(row.context, str) else []
        for _, row in df_candidates.iterrows()
    }

    existing_terms = set(df_locked["term"].astype(str))
    tag_map_step1 = ai_classify_with_context(all_terms, contexts_cache, existing_terms)

    keep_terms = [t for t, tag in tag_map_step1.items() if tag == "Keep"]
    tag_map_step2 = ai_prune_redundant_terms(keep_terms, existing_terms)

    final_tag_map = {}
    for t in all_terms:
        tag = tag_map_step2.get(t, tag_map_step1.get(t, "Need Recheck"))
        final_tag_map[t] = tag

    df_candidates["tag"] = df_candidates["term"].map(final_tag_map)
    df_candidates = df_candidates[df_candidates["tag"] == "Keep"]
    df_candidates = df_candidates[["term", "freq", "context"]]

    df_locked = df_locked[["term", "freq", "context"]]

    df_final = pd.concat([df_locked, df_candidates], ignore_index=True)
    df_final.to_csv(OUTPUT_PHASE3, index=False, encoding="utf-8-sig")
    print(f"✅ Phase 3 done. Wrote {len(df_final)} rows.")
    return df_final

