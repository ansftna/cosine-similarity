"""
extractor.py
============
Modul untuk mengekstrak teks dari file dokumen:
PDF, DOCX, TXT, CSV, dan Excel (XLSX/XLS).

Fungsi utama:
- extract_text_from_pdf()   : Ekstraksi teks dari PDF menggunakan PyMuPDF (fitz)
- extract_text_from_docx()  : Ekstraksi teks dari DOCX menggunakan python-docx
- extract_text_from_txt()   : Ekstraksi teks dari file TXT
- split_text_into_units()   : Memecah teks menjadi unit (per file / per baris)
- process_files_bytes()     : Memproses beberapa file sekaligus (bytes, cache-friendly)
- process_uploaded_files()  : Pembungkus untuk objek file Streamlit

Catatan penting:
- Modul ini HANYA bertanggung jawab atas ekstraksi teks.
- Teks hasil ekstraksi diserahkan ke pipeline preprocessing yang sudah ada.
- Tidak mengubah atau menambahkan logika preprocessing, Binary, atau Cosine Similarity.
"""

import io
import re
from typing import List, Dict, Tuple, Optional


# ─────────────────────────────────────────────
# EKSTRAKSI PDF
# ─────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, str]:
    """
    Mengekstrak teks dari file PDF menggunakan PyMuPDF (fitz).

    Membaca seluruh halaman secara berurutan dan menggabungkan teksnya.
    Jika PDF merupakan scan/image tanpa text layer, fungsi ini akan
    mengembalikan string kosong beserta pesan peringatan.

    Args:
        file_bytes: Konten file PDF dalam bentuk bytes.

    Returns:
        Tuple (text, warning):
        - text   : Teks yang berhasil diekstrak (str).
        - warning: Pesan peringatan jika PDF tidak memiliki teks (str kosong jika tidak ada masalah).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "", "Library PyMuPDF tidak ditemukan. Jalankan: pip install PyMuPDF"

    try:
        # Buka PDF dari bytes
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")  # Ekstrak sebagai plain text
            if page_text.strip():
                pages_text.append(page_text)

        doc.close()
        full_text = "\n".join(pages_text)

        if not full_text.strip():
            warning = (
                "PDF ini tidak memiliki text layer yang dapat diekstrak. "
                "Kemungkinan merupakan PDF hasil scan/image. "
                "Untuk mengekstrak teks dari PDF scan diperlukan OCR (Optical Character Recognition)."
            )
            return "", warning

        return full_text, ""

    except Exception as e:
        return "", f"Gagal membaca PDF: {str(e)}"


# ─────────────────────────────────────────────
# EKSTRAKSI DOCX
# ─────────────────────────────────────────────
def extract_text_from_docx(file_bytes: bytes) -> Tuple[str, str]:
    """
    Mengekstrak teks dari file DOCX menggunakan python-docx.

    Membaca paragraf secara berurutan dan mengabaikan elemen formatting.

    Args:
        file_bytes: Konten file DOCX dalam bentuk bytes.

    Returns:
        Tuple (text, warning):
        - text   : Teks paragraf yang digabungkan dengan newline.
        - warning: Pesan peringatan jika file tidak dapat dibaca (str kosong jika aman).
    """
    try:
        from docx import Document
    except ImportError:
        return "", "Library python-docx tidak ditemukan. Jalankan: pip install python-docx"

    try:
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:  # Abaikan paragraf kosong
                paragraphs.append(text)

        full_text = "\n".join(paragraphs)

        if not full_text.strip():
            return "", "File DOCX tidak memiliki teks yang dapat diekstrak."

        return full_text, ""

    except Exception as e:
        return "", f"Gagal membaca DOCX: {str(e)}"


# ─────────────────────────────────────────────
# EKSTRAKSI TXT
# ─────────────────────────────────────────────
def extract_text_from_txt(file_bytes: bytes) -> Tuple[str, str]:
    """
    Mengekstrak teks dari file TXT.

    Mencoba decoding UTF-8 terlebih dahulu, kemudian fallback ke latin-1.

    Args:
        file_bytes: Konten file TXT dalam bentuk bytes.

    Returns:
        Tuple (text, warning):
        - text   : Konten file sebagai string.
        - warning: Pesan peringatan encoding jika diperlukan (str kosong jika aman).
    """
    # Coba UTF-8 terlebih dahulu
    try:
        text = file_bytes.decode("utf-8")
        if not text.strip():
            return "", "File TXT tidak memiliki konten."
        return text, ""
    except UnicodeDecodeError:
        pass

    # Fallback ke latin-1 (dapat decode semua byte)
    try:
        text = file_bytes.decode("latin-1")
        if not text.strip():
            return "", "File TXT tidak memiliki konten."
        warning = (
            "File TXT tidak dapat dibaca sebagai UTF-8. "
            "Dibaca menggunakan encoding latin-1. "
            "Karakter khusus mungkin tidak ditampilkan dengan benar."
        )
        return text, warning
    except Exception as e:
        return "", f"Gagal membaca file TXT: {str(e)}"


# ─────────────────────────────────────────────
# EKSTRAKSI CSV
# ─────────────────────────────────────────────
def extract_text_from_csv(file_bytes: bytes) -> Tuple[str, str]:
    """
    Mengekstrak teks dari file CSV.

    Setiap baris CSV akan digabung menjadi satu baris teks.
    """

    try:
        import pandas as pd
    except ImportError:
        return "", "Library pandas tidak ditemukan."

    try:
        # Coba UTF-8 terlebih dahulu
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")

        if df.empty:
            return "", "File CSV tidak memiliki data."

        rows = []

        for _, row in df.iterrows():
            values = [
                str(value).strip()
                for value in row.tolist()
                if pd.notna(value) and str(value).strip()
            ]

            if values:
                rows.append(" ".join(values))

        text = "\n".join(rows)

        if not text.strip():
            return "", "File CSV tidak memiliki teks yang dapat diekstrak."

        return text, ""

    except Exception as e:
        return "", f"Gagal membaca CSV: {str(e)}"


# ─────────────────────────────────────────────
# EKSTRAKSI EXCEL
# ─────────────────────────────────────────────
def extract_text_from_excel(file_bytes: bytes) -> Tuple[str, str]:
    """
    Mengekstrak teks dari file Excel XLSX/XLS.

    Setiap baris Excel akan digabung menjadi satu baris teks.
    """

    try:
        import pandas as pd
    except ImportError:
        return "", "Library pandas tidak ditemukan."

    try:
        df = pd.read_excel(io.BytesIO(file_bytes))

        if df.empty:
            return "", "File Excel tidak memiliki data."

        rows = []

        for _, row in df.iterrows():
            values = [
                str(value).strip()
                for value in row.tolist()
                if pd.notna(value) and str(value).strip()
            ]

            if values:
                rows.append(" ".join(values))

        text = "\n".join(rows)

        if not text.strip():
            return "", "File Excel tidak memiliki teks yang dapat diekstrak."

        return text, ""

    except ImportError:
        return "", (
            "Library Excel belum tersedia. "
            "Jalankan: pip install openpyxl"
        )

    except Exception as e:
        return "", f"Gagal membaca Excel: {str(e)}"


# ─────────────────────────────────────────────
# DISPATCH EKSTRAKSI BERDASARKAN EKSTENSI
# ─────────────────────────────────────────────
def extract_text_from_file(filename: str, file_bytes: bytes) -> Tuple[str, str]:
    """
    Menentukan fungsi ekstraksi berdasarkan ekstensi file dan menjalankannya.

    Args:
        filename  : Nama file (digunakan untuk menentukan format).
        file_bytes: Konten file dalam bentuk bytes.

    Returns:
        Tuple (text, warning) dari fungsi ekstraksi yang sesuai.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)

    elif ext == "docx":
        return extract_text_from_docx(file_bytes)

    elif ext == "txt":
        return extract_text_from_txt(file_bytes)

    elif ext == "csv":
        return extract_text_from_csv(file_bytes)

    elif ext in ("xlsx", "xls"):
        return extract_text_from_excel(file_bytes)

    else:
        return "", (
            f"Format file '.{ext}' tidak didukung. "
            "Gunakan PDF, DOCX, TXT, CSV, XLSX, atau XLS."
        )

# ─────────────────────────────────────────────
# PEMISAHAN TEKS MENJADI UNIT
# ─────────────────────────────────────────────
def split_text_into_units(text: str, mode: str = "file") -> List[str]:
    """
    Memecah teks menjadi unit-unit teks berdasarkan mode pemisahan.

    Modes:
    - "file" : Seluruh teks dianggap sebagai SATU unit (default).
    - "line" : Setiap baris non-kosong menjadi satu unit.

    Args:
        text: Teks hasil ekstraksi.
        mode: Mode pemisahan ("file" atau "line").

    Returns:
        Daftar teks/unit yang siap diproses sebagai status.
    """
    if mode == "line":
        lines = text.splitlines()
        # Ambil baris yang tidak kosong setelah strip
        units = [line.strip() for line in lines if line.strip()]
        return units
    else:
        # Mode "file": seluruh teks = satu unit, bersihkan whitespace berlebih
        cleaned = re.sub(r"\n+", " ", text)   # Ganti newline dengan spasi
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return [cleaned] if cleaned else []


# ─────────────────────────────────────────────
# PROSES BEBERAPA FILE SEKALIGUS
# ─────────────────────────────────────────────
def process_files_bytes(
    items: List[Tuple[str, bytes]],
    split_mode: str = "file",
) -> List[Dict]:
    """
    Memproses daftar (nama_file, isi_bytes) dan menghasilkan unit-unit teks.

    Dibuat berbasis bytes agar hasilnya bisa di-cache oleh Streamlit.

    Returns:
        Daftar dict: filename, text, units, warning, char_count, line_count, preview.
    """
    results = []

    for filename, file_bytes in items:
        text, warning = extract_text_from_file(filename, file_bytes)

        non_empty_lines = [l for l in text.splitlines() if l.strip()]
        preview = (text[:120] + "...") if len(text) > 120 else text
        preview = preview.replace("\n", " ")

        units = split_text_into_units(text, mode=split_mode) if text else []

        results.append({
            "filename": filename,
            "text": text,
            "units": units,
            "warning": warning,
            "char_count": len(text),
            "line_count": len(non_empty_lines),
            "preview": preview,
        })

    return results


def process_uploaded_files(uploaded_files: list, split_mode: str = "file") -> List[Dict]:
    """Versi untuk objek file Streamlit (st.file_uploader)."""
    items = []
    for f in uploaded_files:
        data = f.getvalue() if hasattr(f, "getvalue") else f.read()
        items.append((f.name, data))
    return process_files_bytes(items, split_mode)


def collect_all_units(file_results: List[Dict]) -> List[str]:
    """
    Mengumpulkan semua unit teks dari seluruh hasil ekstraksi file.

    Args:
        file_results: Daftar dict hasil process_uploaded_files().

    Returns:
        Daftar teks yang siap dimasukkan ke pipeline run_analysis().
    """
    all_units = []
    for fr in file_results:
        all_units.extend(fr["units"])
    return all_units
