# Cosine Similarity Analyzer

Aplikasi Streamlit untuk menghitung **Cosine Similarity** antar teks dengan **Binary Representation(0/1)**,
mengikuti perhitungan manual pada PDF:

```
sim(x, y) = cos(θ) = (x · y) / (‖x‖ × ‖y‖)
```
Pada vektor binary: `x · y` = jumlah kata yang sama, `‖x‖` = √(jumlah kata unik).

## Fitur
- Input manual (tambah/hapus status) atau upload dokumen **PDF / DOCX / TXT**
- Preprocessing: case folding → hapus tanda baca → tokenisasi
- Tabel binary (kata × status, ada baris *Jumlah* dan daftar kata yang sama pada semua teks)
- Matriks & tabel pasangan cosine similarity + interpretasi, unduh CSV
- Detail perhitungan per pasangan (rumus LaTeX seperti di PDF), pembulatan 3–6 desimal (default 3)
- Tombol **Coba Contoh Data** memuat 5 status dari PDF

## Menjalankan lokal
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud
1. Upload isi folder ini ke repository GitHub (`app.py` berada di root repo).
2. Buka https://share.streamlit.io → **New app** → pilih repo & branch.
3. **Main file path**: `app.py` → **Deploy**.

## Struktur
```
app.py                 # antarmuka Streamlit
assets/style.css       # tema (palet #D852E8 #712FDA #F9218D #F4FEFF #707BAA)
.streamlit/config.toml # tema & pengaturan server
core/preprocessing.py  # case folding, hapus tanda baca, tokenisasi, vocabulary
core/similarity.py     # binary, cosine, interpretasi, detail pasangan
core/pipeline.py       # run_analysis() — menggabungkan semua tahap
core/extractor.py      # ekstraksi teks PDF/DOCX/TXT
tests/test_core.py     # 20 tes:  python -m unittest discover -s tests -v
sample_data/           # contoh 5 status dari PDF (satu baris = satu status)
```

## Catatan tentang PDF
Dua hal pada perhitungan tulisan tangan di PDF:
1. **Jumlah kata S5** ditulis 6, padahal "saya suka makan sayur kangkung" hanya **5 kata** (kolom S5 pada
   tabel binary pun hanya berisi lima angka 1). Sistem menghitung dengan benar (5), sehingga 4 pasangan S5 berbeda:

   | Pasangan | PDF | Sistem |
   |---|---|---|
   | S1–S5 | 0.548 | **0.600** (3/√25 = 3/5) |
   | S2–S5 | 0.500 | **0.548** (3/√30) |
   | S3–S5 | 0.433 | **0.474** (3/√40) |
   | S4–S5 | 0.500 | **0.548** (3/√30) |

   Enam pasangan lain (S1–S2, S1–S3, S1–S4, S2–S3, S2–S4, S3–S4) sama persis dengan PDF.
2. Penyederhanaan `3/(√6×√6)` ditulis `3/√6` di PDF; yang benar `3/6 = 0.500`. Aplikasi menampilkan bentuk `3/6`.
