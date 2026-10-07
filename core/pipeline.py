"""
pipeline.py
===========
Menggabungkan seluruh tahap: preprocessing → vocabulary → binary → cosine.
Dipisah dari app.py agar mudah diuji tanpa Streamlit.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .preprocessing import build_vocabulary, preprocess_all
from .similarity import (
    build_binary_matrix,
    build_pair_details,
    build_similarity_matrix,
    common_words_all,
    get_similarity_pairs,
)


def run_analysis(
    texts: List[str],
    labels: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> Dict:
    """
    Jalankan analisis kemiripan.

    Args:
        texts   : daftar teks (boleh ada yang kosong; yang kosong dilewati).
        labels  : label tiap teks (default S1, S2, ...), sejajar dengan `texts`.
        sources : keterangan sumber tiap teks (mis. nama file), sejajar dengan `texts`.

    Raises:
        ValueError bila teks tidak kosong kurang dari 2.
    """
    keep = [i for i, t in enumerate(texts) if t and t.strip()]
    if len(keep) < 2:
        raise ValueError("Minimal diperlukan 2 teks yang tidak kosong.")

    valid = [texts[i] for i in keep]
    ids = [labels[i] for i in keep] if labels else [f"S{k + 1}" for k in range(len(keep))]
    srcs = [sources[i] for i in keep] if sources else None

    preprocessed = preprocess_all(valid)
    vocabulary = build_vocabulary(preprocessed)
    binary_matrix = build_binary_matrix(preprocessed, vocabulary)
    sim_matrix = build_similarity_matrix(binary_matrix)
    pairs_df = get_similarity_pairs(sim_matrix, ids)
    details = build_pair_details(preprocessed, ids)

    # Tabel preprocessing
    prep_rows = []
    for k, (sid, res) in enumerate(zip(ids, preprocessed)):
        row = {"ID": sid}
        if srcs:
            row["Sumber"] = srcs[k]
        row.update(
            {
                "Teks Asli": res["original"],
                "Case Folding": res["case_folded"],
                "Hapus Tanda Baca": res["no_punctuation"],
                "Token": ", ".join(res["tokens"]) if res["tokens"] else "(kosong)",
                "Jumlah Kata Unik": len(set(res["tokens"])),
            }
        )
        prep_rows.append(row)
    prep_df = pd.DataFrame(prep_rows)

    # Tabel binary: baris = kata, kolom = status (seperti tabel manual)
    binary_df = pd.DataFrame(binary_matrix.T.astype(int), index=vocabulary, columns=ids)
    binary_df.index.name = "Kata"
    binary_total = binary_df.sum(axis=0)

    sim_df = pd.DataFrame(sim_matrix, index=ids, columns=ids)

    values = pairs_df["Cosine Similarity"]
    summary = {
        "jumlah_status": len(ids),
        "jumlah_vocabulary": len(vocabulary),
        "jumlah_token": int(sum(len(r["tokens"]) for r in preprocessed)),
        "similarity_tertinggi": float(values.max()),
        "similarity_terendah": float(values.min()),
        "pasangan_tertinggi": " – ".join(pairs_df.loc[values.idxmax(), ["Status 1", "Status 2"]]),
        "pasangan_terendah": " – ".join(pairs_df.loc[values.idxmin(), ["Status 1", "Status 2"]]),
    }

    return {
        "source": tuple(t.strip() for t in valid),
        "status_ids": ids,
        "preprocessed": preprocessed,
        "vocabulary": vocabulary,
        "binary_matrix": binary_matrix,
        "sim_matrix": sim_matrix,
        "prep_df": prep_df,
        "binary_df": binary_df,
        "binary_total": binary_total,
        "sim_df": sim_df,
        "pairs_df": pairs_df,
        "details": details,
        "common_words": common_words_all(preprocessed, vocabulary),
        "summary": summary,
    }
