"""
app.py
======
Cosine Similarity Analyzer — aplikasi Streamlit.

Alur (sesuai perhitungan manual pada PDF):
    Teks → Preprocessing (case folding, hapus tanda baca, tokenisasi)
         → Representasi Binary (0/1)
         → Cosine Similarity:  sim(x, y) = cos(θ) = (x · y) / (‖x‖ × ‖y‖)

Sumber data: input manual atau upload dokumen (PDF / DOCX / TXT).
Jalankan lokal :  streamlit run app.py
"""
    
from __future__ import annotations

import html as html_lib
import inspect
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

from core.extractor import process_files_bytes
from core.pipeline import run_analysis

BASE_DIR = Path(__file__).parent

# Contoh data = lima status pada PDF perhitungan manual
EXAMPLE_TEXTS = [
    "Saya suka sayur dan buah",
    "Saya suka sayur tapi tak suka kimchi",
    "Hampir semua sayur saya suka. Pare pun doyan",
    "Saya suka sayur sayuran di Tesco.",
    "Saya suka makan sayur kangkung",
]
MODE_MANUAL = "Input Manual"
MODE_UPLOAD = "Upload Dokumen"

st.set_page_config(
    page_title="Cosine Similarity Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────
# UTILITAS TAMPILAN
# ─────────────────────────────────────────────
def _load_css() -> None:
    css = (BASE_DIR / "assets" / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render HTML. Baris dirapatkan agar tidak dianggap blok kode oleh Markdown."""
    compact = " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(compact, unsafe_allow_html=True)


# Kompatibilitas lintas versi Streamlit (use_container_width → width="stretch")
_BTN_STRETCH = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.button).parameters
    else {"use_container_width": True}
)


_DL_STRETCH = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.download_button).parameters
    else {"use_container_width": True}
)


def show_df(data, **kwargs) -> None:
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except Exception:  # Streamlit versi lama
        st.dataframe(data, use_container_width=True, **kwargs)

def show_preprocessing_table(data) -> None:
    headers = list(data.columns)

    header_html = "".join(
        f"<th>{html_lib.escape(str(col))}</th>"
        for col in headers
    )

    rows_html = []

    for _, row in data.iterrows():
        cells = []

        for col in headers:
            value = row[col]

            if col == "Token":
                if value == "(kosong)":
                    cell = '<span>(kosong)</span>'
                else:
                    tokens = [x.strip() for x in str(value).split(",") if x.strip()]

                    cell = "".join(
                        f'<span class="token-bubble">{html_lib.escape(token)}</span>'
                        for token in tokens
                    )

                cells.append(f"<td>{cell}</td>")

            else:
                cells.append(
                    f"<td>{html_lib.escape(str(value))}</td>"
                )

        rows_html.append(
            "<tr>" + "".join(cells) + "</tr>"
        )

    html(
        f"""
        <div class="preprocessing-table">
            <table>
                <thead>
                    <tr>{header_html}</tr>
                </thead>
                <tbody>
                    {"".join(rows_html)}
                </tbody>
            </table>
        </div>
        """
    )
def section(title: str, subtitle: str = "") -> None:
    sub = f'<p class="sec-sub">{subtitle}</p>' if subtitle else ""
    html(f'<div class="sec"><div class="sec-title">{title}</div>{sub}</div>')


def note(text: str, kind: str = "") -> None:
    html(f'<div class="note {kind}">{text}</div>')


def empty_state(icon: str, message: str) -> None:
    html(f'<div class="empty"><div class="ico">{icon}</div>{message}</div>')


def csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode("utf-8-sig")


def fmt(value: float) -> str:
    return f"{value:.{st.session_state.decimals}f}"


def tex_label(status_id: str) -> str:
    m = re.fullmatch(r"S(\d+)", status_id)
    return f"S_{{{m.group(1)}}}" if m else rf"\text{{{status_id}}}"


# ─────────────────────────────────────────────
# STATE & CALLBACK
# ─────────────────────────────────────────────
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("n_status", 2)
    ss.setdefault("results", None)
    ss.setdefault("decimals", 3)  # PDF memakai 3 angka di belakang koma
    ss.setdefault("input_mode", MODE_MANUAL)


def manual_texts() -> List[str]:
    return [st.session_state.get(f"status_{i}", "") for i in range(st.session_state.n_status)]


def _set_texts(texts: List[str]) -> None:
    st.session_state.n_status = max(2, len(texts))
    for i in range(st.session_state.n_status):
        st.session_state[f"status_{i}"] = texts[i] if i < len(texts) else ""


def cb_add() -> None:
    st.session_state.n_status += 1


def cb_remove(idx: int) -> None:
    ss = st.session_state
    n = ss.n_status
    for k in range(idx, n - 1):
        ss[f"status_{k}"] = ss.get(f"status_{k + 1}", "")
    ss.n_status = n - 1


def cb_reset() -> None:
    _set_texts(["", ""])
    st.session_state.results = None
    st.session_state.input_mode = MODE_MANUAL


def cb_example() -> None:
    """Muat lima status dari PDF dan langsung jalankan analisis."""
    _set_texts(EXAMPLE_TEXTS)
    st.session_state.input_mode = MODE_MANUAL
    st.session_state.decimals = 3
    labels = [f"S{i + 1}" for i in range(len(EXAMPLE_TEXTS))]
    st.session_state.results = run_analysis(EXAMPLE_TEXTS, labels=labels)


def cb_analyze_manual() -> None:
    """Dibaca saat callback berjalan, sehingga selalu memakai teks terbaru."""
    raw = manual_texts()
    labels = [f"S{i + 1}" for i in range(len(raw))]
    st.session_state.results = run_analysis(raw, labels=labels)


def cb_analyze_upload(units: List[str], labels: List[str], sources: List[str]) -> None:
    st.session_state.results = run_analysis(units, labels=labels, sources=sources)


@st.cache_data(show_spinner=False)
def example_pairs() -> list:
    res = run_analysis(EXAMPLE_TEXTS)
    return [
        (f"{r['Status 1']} – {r['Status 2']}", float(r["Cosine Similarity"]))
        for _, r in res["pairs_df"].iterrows()
    ]


@st.cache_data(show_spinner=False)
def extract_cached(items: tuple, split_mode: str) -> list:
    return process_files_bytes(list(items), split_mode)


init_state()
_load_css()


# ─────────────────────────────────────────────
# BRAND BAR & HERO
# ─────────────────────────────────────────────
html(
    """
    <div class="brandbar">
      <div class="brand"><div class="brand-logo">CS</div><span class="brand-name">Cosine Similarity</span></div>
      <span class="brand-tag">Text Mining · Binary Representation · Cosine Similarity</span>
    </div>
    """
)


def hero_visual() -> str:
    res = st.session_state.results
    if res is not None:
        df = res["pairs_df"].sort_values("Cosine Similarity", ascending=False).head(4)
        pairs = [(f"{r['Status 1']} – {r['Status 2']}", float(r["Cosine Similarity"])) for _, r in df.iterrows()]
        tag, foot = f"{len(res['details'])} pasangan", "Hasil analisis Anda"
    else:
        pairs = example_pairs()[:4]
        tag, foot = "Contoh", "Contoh Data · klik “Coba Contoh Data”"
    rows = "".join(
        f'<div class="pair-row"><span class="pair-name">{html_lib.escape(name)}</span>'
        f'<span class="pair-val">{fmt(val)}</span></div>'
        f'<div class="bar"><i style="width:{val * 100:.1f}%"></i></div>'
        for name, val in pairs
    )
    return (
        '<div class="hero-visual"><div class="blob blob-pink"></div><div class="blob blob-violet"></div>'
        '<div class="blob blob-dot"></div><div class="phone">'
        f'<div class="phone-head"><span class="phone-title">Cosine Similarity</span><span class="phone-tag">{tag}</span></div>'
        '<div class="phone-formula">cos(θ) = x · y / ( ‖x‖ × ‖y‖ )</div>'
        f"{rows}<div class='phone-foot'>{foot}</div></div></div>"
    )


hero_left, hero_right = st.columns([1.1, 1], gap="large")
with hero_left:
    html(
        """
        <div class="eyebrow">Analisis kemiripan teks</div>
        <div class="hero-title">Hitung kemiripan teks dengan Cosine Similarity</div>
        <p class="hero-desc">Masukkan beberapa kalimat atau unggah dokumen. Sistem melakukan preprocessing,
        membentuk binary representation, lalu menghitung cosine similarity setiap pasangan
        lengkap dengan langkah perhitungannya.</p>
        """
    )
    b1, b2, _sp = st.columns([1.5, 0.9, 1.1])
    with b1:
        st.button("Coba Contoh Data", type="primary", key="hero_example", on_click=cb_example, **_BTN_STRETCH)
    with b2:
        st.button("Reset", key="hero_reset", on_click=cb_reset, **_BTN_STRETCH)
with hero_right:
    html(hero_visual())

st.write("")

tab_input, tab_prep, tab_binary, tab_sim, tab_detail = st.tabs(
    ["Input", "Preprocessing", "Binary Representation", "Similarity", "Detail Perhitungan"]
)

# ══════════════════════════════════════════════
# TAB INPUT
# ══════════════════════════════════════════════
units: List[str] = []
labels: Optional[List[str]] = None
sources: Optional[List[str]] = None

with tab_input:
    section("Sumber Data", "Pilih cara memasukkan teks yang ingin dibandingkan.")
    mode = st.radio(
        "Sumber data",
        [MODE_MANUAL, MODE_UPLOAD],
        horizontal=True,
        key="input_mode",
        label_visibility="collapsed",
    )

# ── Input manual ──
    if mode == MODE_MANUAL:
        c1, c2, _ = st.columns([1.2, 1.4, 3])
        with c1:
            st.button("＋ Tambah Status", key="add_status", on_click=cb_add, **_BTN_STRETCH)
        with c2:
            st.button("Muat Contoh Data", key="load_example", on_click=cb_example, **_BTN_STRETCH)

        n = st.session_state.n_status
        for i in range(n):
            col_badge, col_text, col_del = st.columns([0.7, 10, 0.9], vertical_alignment="center")
            with col_badge:
                html(f'<div class="badge">S{i + 1}</div>')
            with col_text:
                st.text_area(
                    f"Status S{i + 1}",
                    key=f"status_{i}",
                    height=78,
                    placeholder=f"Tulis teks S{i + 1} di sini…",
                    label_visibility="collapsed",
                )
            with col_del:
                st.button("✕", key=f"del_{i}", on_click=cb_remove, args=(i,), disabled=n <= 2, help=f"Hapus S{i + 1}")

        raw = manual_texts()
        units = raw
        labels = [f"S{i + 1}" for i in range(len(raw))]
        valid_count = sum(1 for t in raw if t.strip())
        if valid_count == 0:
            note("Isi teks di atas, atau klik <b>Muat Contoh Data</b> untuk memakai data contoh.")
        elif valid_count == 1:
            note("Minimal diperlukan <b>2 teks</b> yang tidak kosong untuk dibandingkan.", "warn")
        else:
            note(f"<b>{valid_count} teks</b> siap dianalisis.", "ok")
        can_analyze = valid_count >= 2

    # ── Upload dokumen ──
    else:
        note("Unggah satu atau beberapa file "
             "<b>PDF</b>, <b>DOCX</b>, <b>TXT</b>, "
             "<b>CSV</b>, atau <b>Excel</b>. "
             "Teks diekstrak otomatis."
            )
        files = st.file_uploader(
            "Upload dokumen",
            type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        can_analyze = False
        if files:
            split_opt = st.radio(
                "Pemisahan teks",
                ["Satu file = satu teks", "Satu baris = satu teks"],
                horizontal=True,
            )
            split_mode = "file" if split_opt.startswith("Satu file") else "line"
            items = tuple((f.name, f.getvalue()) for f in files)
            with st.spinner("Mengekstrak teks…"):
                file_results = extract_cached(items, split_mode)

            show_df(
                pd.DataFrame(
                    [
                        {
                            "File": fr["filename"],
                            "Karakter": fr["char_count"],
                            "Baris": fr["line_count"],
                            "Unit Teks": len(fr["units"]),
                            "Preview": fr["preview"] or "(kosong)",
                        }
                        for fr in file_results
                    ]
                ),
                hide_index=True,
            )
            for fr in file_results:
                if fr["warning"]:
                    note(f"<b>{html_lib.escape(fr['filename'])}</b>: {html_lib.escape(fr['warning'])}", "warn")

            units, labels, sources, ext_rows = [], [], [], []
            for fr in file_results:
                for k, unit in enumerate(fr["units"], start=1):
                    if unit.strip():
                        units.append(unit)
                        labels.append(f"S{len(units)}")
                        sources.append(fr["filename"] if len(fr["units"]) == 1 else f"{fr['filename']} (baris {k})")
                        ext_rows.append({"File": fr["filename"], "ID": f"S{len(units)}", "Teks": unit})

            if len(units) < 2:
                note("Teks yang berhasil diekstrak kurang dari <b>2</b>. Tambahkan dokumen atau ubah pemisahan teks.", "warn")
            else:
                note(f"<b>{len(units)} teks</b> berhasil diekstrak dan siap dianalisis.", "ok")
                can_analyze = True
                st.download_button(
                    "Unduh Teks Hasil Ekstraksi (CSV)",
                    csv_bytes(pd.DataFrame(ext_rows)),
                    "teks_hasil_ekstraksi.csv",
                    "text/csv",
                    key="dl_extracted",
                )
        else:
            empty_state("📂", "Belum ada dokumen yang diunggah.")

    # ── Tombol analisis ──
    st.write("")
    a1, a2, _ = st.columns([1.5, 1, 2.6], vertical_alignment="bottom")
    with a1:
        if mode == MODE_MANUAL:
            analyze_kwargs = {"on_click": cb_analyze_manual}
        else:
            analyze_kwargs = {"on_click": cb_analyze_upload, "args": (units, labels, sources)}
        st.button(
            "Analisis Kemiripan",
            type="primary",
            key="analyze",
            disabled=not can_analyze,
            **analyze_kwargs,
            **_BTN_STRETCH,
        )
    with a2:
        st.selectbox("Angka di belakang koma", [3, 4, 5, 6], key="decimals")

    # ── Ringkasan ──
    res = st.session_state.results
    if res is not None:
        sm = res["summary"]
        section("Ringkasan Hasil")
        cards = [
            (sm["jumlah_status"], "Jumlah Teks", ""),
            (sm["jumlah_vocabulary"], "Kata Unik", ""),
            (sm["jumlah_token"], "Total Kata", ""),
            (fmt(sm["similarity_tertinggi"]), "Tertinggi", sm["pasangan_tertinggi"]),
            (fmt(sm["similarity_terendah"]), "Terendah", sm["pasangan_terendah"]),
        ]
        for col, (value, label, sub) in zip(st.columns(5), cards):
            with col:
                sub_html = f'<div class="s">{html_lib.escape(sub)}</div>' if sub else ""
                html(f'<div class="metric"><div class="v">{value}</div><div class="l">{label}</div>{sub_html}</div>')
        note("Buka tab <b>Preprocessing</b>, <b>Binary</b>, <b>Similarity</b>, dan <b>Detail Perhitungan</b> untuk melihat setiap tahapnya.")

res = st.session_state.results
current_source = tuple(t.strip() for t in units if t and t.strip())
stale = res is not None and res["source"] != current_source


def stale_note() -> None:
    if stale:
        note("Teks pada tab Input berbeda dari hasil yang ditampilkan. Klik <b>Analisis Kemiripan</b> untuk memperbarui.", "warn")


def need_analysis() -> None:
    empty_state("📊", "Belum ada hasil. Isi data pada tab <b>Input</b> lalu klik <b>Analisis Kemiripan</b>, atau klik <b>Coba Contoh Data</b>.")


# ══════════════════════════════════════════════
# TAB PREPROCESSING
# ══════════════════════════════════════════════
with tab_prep:
    section("Preprocessing", "Teks dibersihkan sebelum dibandingkan.")
    if res is None:
        need_analysis()
    else:
        stale_note()
        html(
            '<div class="steps"><span class="step">1 · Case Folding</span><span class="step-arrow">→</span>'
            '<span class="step">2 · Hapus Tanda Baca</span><span class="step-arrow">→</span>'
            '<span class="step">3 · Tokenisasi</span></div>'
        )
        show_preprocessing_table(res["prep_df"])
        note(
            "Pada representasi binary, kata yang berulang dalam satu teks hanya dihitung <b>sekali</b> "
            "(kolom <i>Jumlah Kata Unik</i>). Contoh: “suka” muncul dua kali di S2, tetapi dihitung satu."
        )
        st.download_button("Unduh Preprocessing (CSV)", csv_bytes(res["prep_df"]), "preprocessing.csv", "text/csv", key="dl_prep")


# ══════════════════════════════════════════════
# TAB BINARY
# ══════════════════════════════════════════════
def style_binary(df: pd.DataFrame) -> pd.DataFrame:
    zero = "color:#B4BAD6;text-align:center;"
    one = "background-color:rgba(216,82,232,0.22);color:#712FDA;font-weight:700;text-align:center;"
    total = "background-color:#F4FEFF;color:#712FDA;font-weight:700;text-align:center;"
    out = pd.DataFrame(zero, index=df.index, columns=df.columns)
    out = out.mask(df == 1, one)
    out.loc["Jumlah"] = total
    return out


with tab_binary:
    section("Binary Representation (0/1)", "Satu jika kata muncul pada teks, nol jika tidak.")
    if res is None:
        need_analysis()
    else:
        stale_note()
        table = pd.concat([res["binary_df"], pd.DataFrame([res["binary_total"]], index=["Jumlah"])])
        table.index.name = "Kata"
        show_df(
            table.style.apply(style_binary, axis=None),
            height=min(38 * (len(table) + 1) + 3, 640),
        )
        common = res["common_words"]
        if common:
            chips = "".join(f'<span class="chip">{html_lib.escape(w)}</span>' for w in common)
            html(f'<div class="note ok"><b>Kata yang sama pada semua teks ({len(common)}):</b><br>{chips}</div>')
        else:
            note("Tidak ada kata yang muncul pada <b>semua</b> teks.", "warn")
        st.download_button(
            "Unduh Tabel Binary (CSV)",
            csv_bytes(table, index=True),
            "representasi_binary.csv",
            "text/csv",
            key="dl_binary",
        )


# ══════════════════════════════════════════════
# TAB SIMILARITY
# ══════════════════════════════════════════════
def style_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame("", index=df.index, columns=df.columns)
    for i, r in enumerate(df.index):
        for j, c in enumerate(df.columns):
            v = float(df.loc[r, c])
            if i == j:
                out.loc[r, c] = "background-color:#712FDA;color:#fff;font-weight:700;text-align:center;"
            else:
                alpha = 0.06 + 0.5 * v
                out.loc[r, c] = f"background-color:rgba(216,82,232,{alpha:.2f});color:#2B1760;font-weight:600;text-align:center;"
    return out


with tab_sim:
    section("Cosine Similarity", "Nilai 0 berarti tidak ada kata yang sama, nilai 1 berarti kata-katanya identik.")
    if res is None:
        need_analysis()
    else:
        stale_note()
        dec = st.session_state.decimals
        st.markdown("#### Matriks Cosine Similarity")
        show_df(res["sim_df"].style.apply(style_matrix, axis=None).format(f"{{:.{dec}f}}"))
        html(
            '<div class="legend"><span style="background:#fff">Sangat Rendah &lt; 0.20</span>'
            '<span style="background:#FBEAFD">Rendah 0.20–0.39</span><span style="background:#F5CFFA">Sedang 0.40–0.59</span>'
            '<span style="background:#EEB2F6">Tinggi 0.60–0.79</span><span style="background:#E495F1">Sangat Tinggi ≥ 0.80</span></div>'
        )

        st.markdown("#### Pasangan Teks")
        pairs = res["pairs_df"]
        show_df(
            pairs,
            hide_index=True,
            column_config={
                "Cosine Similarity": st.column_config.ProgressColumn(
                    "Cosine Similarity", min_value=0.0, max_value=1.0, format=f"%.{dec}f"
                ),
            },
        )

        st.markdown("#### Unduh Hasil")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button("Preprocessing", csv_bytes(res["prep_df"]), "preprocessing.csv", "text/csv", key="dl_p2", **_DL_STRETCH)
        with d2:
            st.download_button("Tabel Binary", csv_bytes(res["binary_df"], index=True), "representasi_binary.csv", "text/csv", key="dl_b2", **_DL_STRETCH)
        with d3:
            st.download_button("Matriks Similarity", csv_bytes(res["sim_df"].round(dec), index=True), "matriks_similarity.csv", "text/csv", key="dl_m2", **_DL_STRETCH)
        with d4:
            st.download_button("Pasangan Teks", csv_bytes(pairs.round({"Cosine Similarity": dec})), "pasangan_similarity.csv", "text/csv", key="dl_pr2", **_DL_STRETCH)


# ══════════════════════════════════════════════
# TAB DETAIL PERHITUNGAN
# ══════════════════════════════════════════════
def latex_expression(d: dict, dec: int) -> str:
    a, b, s = d["count_a"], d["count_b"], d["shared_count"]
    lhs = rf"\cos({tex_label(d['id_a'])},{tex_label(d['id_b'])})"
    val = f"{d['similarity']:.{dec}f}"
    if a == 0 or b == 0:
        return rf"{lhs} = 0 \;\; \text{{(teks kosong setelah preprocessing)}}"
    if s == 0:
        return rf"{lhs} = \frac{{0}}{{\sqrt{{{a}}} \times \sqrt{{{b}}}}} = 0"
    if a == b:
        return rf"{lhs} = \frac{{{s}}}{{\sqrt{{{a}}} \times \sqrt{{{a}}}}} = \frac{{{s}}}{{{a}}} = {val}"
    return rf"{lhs} = \frac{{{s}}}{{\sqrt{{{a}}} \times \sqrt{{{b}}}}} = \frac{{{s}}}{{\sqrt{{{a * b}}}}} = {val}"


with tab_detail:
    section("Detail Perhitungan", "Langkah perhitungan manual untuk setiap pasangan teks.")
    if res is None:
        need_analysis()
    else:
        stale_note()
        dec = st.session_state.decimals
        st.latex(r"sim(x,y) = \cos(\theta) = \frac{x \cdot y}{\|x\| \, \|y\|}")
        note(
            "Pada vektor binary: <b>x · y</b> = jumlah kata yang sama pada kedua teks, dan "
            "<b>‖x‖</b> = √(jumlah kata unik teks tersebut)."
        )
        details = res["details"]
        expand_all = len(details) <= 10

        for number, d in enumerate(details, start=1):
            title = f"{number}. {d['id_a']} dan {d['id_b']}  —  {d['similarity']:.{dec}f}"
            if expand_all:
                box = st.container(border=True)
                with box:
                    html(
                        f'<div class="pair-head"><div class="pair-title"><span class="num">{number}</span>'
                        f"{html_lib.escape(d['id_a'])} dan {html_lib.escape(d['id_b'])}</div>"
                        f'<span class="score">{d["similarity"]:.{dec}f}</span></div>'
                    )
            else:
                box = st.expander(title)
            with box:
                left, right = st.columns([1, 1.5])
                with left:
                    words = ", ".join(d["shared_words"]) if d["shared_words"] else "tidak ada"
                    html(
                        f'<div class="kv">Jumlah kata <b>{html_lib.escape(d["id_a"])}</b> = <b>{d["count_a"]}</b><br>'
                        f'Jumlah kata <b>{html_lib.escape(d["id_b"])}</b> = <b>{d["count_b"]}</b><br>'
                        f'Kata yang sama (<b>{d["shared_count"]}</b>): {html_lib.escape(words)}</div>'
                    )
                with right:
                    st.latex(latex_expression(d, dec))

html('<div class="footer">Cosine Similarity · Binary Representation. anasiskaf</div>')
