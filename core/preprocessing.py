"""
preprocessing.py
================
Tahapan preprocessing sebelum perhitungan Cosine Similarity:

1. Case Folding     : huruf menjadi kecil semua
2. Hapus Tanda Baca : seluruh tanda baca dihapus
3. Tokenisasi       : teks dipecah menjadi daftar kata

Tidak ada stemming / stopword removal, sehingga "sayur" dan "sayuran"
diperlakukan sebagai dua kata yang berbeda (sama seperti perhitungan manual).
"""

import re
from typing import Dict, List

# Semua karakter yang bukan huruf/angka/spasi dianggap tanda baca.
# (Mencakup string.punctuation, "_" serta tanda baca unicode seperti “ ” ‘ ’ – …)
_PUNCTUATION_RE = re.compile(r"[^\w\s]|_", flags=re.UNICODE)


def case_folding(text: str) -> str:
    """Ubah seluruh huruf menjadi huruf kecil."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Hapus tanda baca dan rapikan spasi berlebih."""
    cleaned = _PUNCTUATION_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize(text: str) -> List[str]:
    """Pecah teks bersih menjadi token. Teks kosong menghasilkan list kosong."""
    if not text.strip():
        return []
    return text.strip().split()


def preprocess_text(text: str) -> Dict[str, object]:
    """Jalankan seluruh pipeline preprocessing pada satu teks."""
    original = text.strip()
    case_folded = case_folding(original)
    no_punct = remove_punctuation(case_folded)
    tokens = tokenize(no_punct)
    return {
        "original": original,
        "case_folded": case_folded,
        "no_punctuation": no_punct,
        "tokens": tokens,
    }


def preprocess_all(texts: List[str]) -> List[Dict[str, object]]:
    """Jalankan preprocessing pada seluruh daftar teks."""
    return [preprocess_text(t) for t in texts]


def build_vocabulary(preprocessed_results: List[Dict], sort: bool = False) -> List[str]:
    """
    Bangun vocabulary (kata unik) dari seluruh teks.

    Secara default urutan mengikuti kemunculan pertama kata (S1 → S2 → ...),
    persis seperti tabel kata pada perhitungan manual. Gunakan sort=True
    untuk urutan alfabetis.
    """
    vocab: Dict[str, None] = {}
    for result in preprocessed_results:
        for token in result["tokens"]:
            vocab.setdefault(token, None)
    words = list(vocab.keys())
    return sorted(words) if sort else words
