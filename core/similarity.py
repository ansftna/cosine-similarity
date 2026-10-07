"""
similarity.py
=============
Representasi  (0/1) dan Cosine Similarity.

Rumus (sesuai perhitungan manual):

    sim(x, y) = cos(θ) = (x · y) / (‖x‖ × ‖y‖)

Pada representasi :
    x · y = jumlah kata yang sama pada kedua teks
    ‖x‖   = √(jumlah kata unik teks x)
sehingga   cos(x, y) = kata_sama / (√jumlah_x × √jumlah_y)
"""

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# BINARY REPRESENTATION
# ─────────────────────────────────────────────
def build_binary_vector(tokens: List[str], vocabulary: List[str]) -> np.ndarray:
    """1 jika kata ada pada teks, 0 jika tidak."""
    token_set = set(tokens)
    return np.array([1 if w in token_set else 0 for w in vocabulary], dtype=float)


def build_binary_matrix(preprocessed_results: List[Dict], vocabulary: List[str]) -> np.ndarray:
    """Matrix  (n_teks × n_vocabulary); tiap baris = vektor satu teks."""
    return np.array(
        [build_binary_vector(r["tokens"], vocabulary) for r in preprocessed_results]
    ).reshape(len(preprocessed_results), len(vocabulary))


# ─────────────────────────────────────────────
# COSINE SIMILARITY
# ─────────────────────────────────────────────
def cosine_similarity_pair(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """cos(A, B) = (A · B) / (‖A‖ × ‖B‖); 0.0 bila salah satu vektor kosong."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), 0.0, 1.0))


def cosine_from_counts(shared: int, count_a: int, count_b: int) -> float:
    """Bentuk ringkas untuk vektor : kata_sama / (√a × √b)."""
    if count_a == 0 or count_b == 0:
        return 0.0
    return shared / (math.sqrt(count_a) * math.sqrt(count_b))


def build_similarity_matrix(binary_matrix: np.ndarray) -> np.ndarray:
    """Matriks cosine (n × n): simetris, diagonal = 1 (kecuali teks kosong)."""
    n = len(binary_matrix)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i][j] = cosine_similarity_pair(binary_matrix[i], binary_matrix[j])
    return sim


def interpret_similarity(value: float) -> str:
    """Label interpretasi nilai cosine similarity."""
    value = round(value, 9)  # hindari galat floating point di batas kategori (0.6 → 0.5999…)
    if value >= 0.80:
        return "Sangat Tinggi"
    if value >= 0.60:
        return "Tinggi"
    if value >= 0.40:
        return "Sedang"
    if value >= 0.20:
        return "Rendah"
    return "Sangat Rendah"


def get_similarity_pairs(
    sim_matrix: np.ndarray,
    status_ids: List[str],
    decimals: Optional[int] = None,
) -> pd.DataFrame:
    """Tabel pasangan unik (i < j) beserta nilai dan interpretasinya."""
    rows = []
    n = len(status_ids)
    for i in range(n):
        for j in range(i + 1, n):
            value = float(sim_matrix[i][j])
            rows.append(
                {
                    "Status 1": status_ids[i],
                    "Status 2": status_ids[j],
                    "Cosine Similarity": round(value, decimals) if decimals is not None else value,
                    "Interpretasi": interpret_similarity(value),
                }
            )
    return pd.DataFrame(rows, columns=["Status 1", "Status 2", "Cosine Similarity", "Interpretasi"])


# ─────────────────────────────────────────────
# DETAIL PERHITUNGAN
# ─────────────────────────────────────────────
def common_words_all(preprocessed_results: List[Dict], vocabulary: List[str]) -> List[str]:
    """Kata yang muncul pada SEMUA teks (urutan mengikuti vocabulary)."""
    if not preprocessed_results:
        return []
    token_sets = [set(r["tokens"]) for r in preprocessed_results]
    shared = set.intersection(*token_sets)
    return [w for w in vocabulary if w in shared]


def build_pair_details(preprocessed_results: List[Dict], status_ids: List[str]) -> List[Dict]:
    """
    Komponen perhitungan manual untuk setiap pasangan (i < j):
    jumlah kata unik masing-masing, kata yang sama, dan nilai cosine.
    """
    details = []
    n = len(status_ids)
    for i in range(n):
        for j in range(i + 1, n):
            set_a = set(preprocessed_results[i]["tokens"])
            set_b = set(preprocessed_results[j]["tokens"])
            shared_words = [w for w in preprocessed_results[i]["tokens"] if w in set_b]
            shared_words = list(dict.fromkeys(shared_words))  # unik, urutan teks A
            details.append(
                {
                    "id_a": status_ids[i],
                    "id_b": status_ids[j],
                    "count_a": len(set_a),
                    "count_b": len(set_b),
                    "shared_words": shared_words,
                    "shared_count": len(shared_words),
                    "similarity": cosine_from_counts(len(shared_words), len(set_a), len(set_b)),
                }
            )
    return details
