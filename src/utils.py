import pandas as pd

from config import (
    MODEL_DIR,
    OUTPUT_DIR,
    VISUAL_DIR,
    VISUAL_BEFORE_DIR,
    VISUAL_AFTER_DIR,
    VISUAL_AUDIT_DIR,
    VISUAL_EVALUATION_DIR,
    VISUAL_FEATURE_IMPORTANCE_DIR,
)


def create_project_directories():
    """
    Membuat seluruh folder output yang dibutuhkan project.
    Fungsi ini aman dijalankan berulang kali.
    """
    directories = [
        MODEL_DIR,
        OUTPUT_DIR,
        VISUAL_DIR,
        VISUAL_BEFORE_DIR,
        VISUAL_AFTER_DIR,
        VISUAL_AUDIT_DIR,
        VISUAL_EVALUATION_DIR,
        VISUAL_FEATURE_IMPORTANCE_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_dataframe(dataframe: pd.DataFrame, path):
    """
    Menyimpan DataFrame ke CSV.
    """
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")


def print_section(title: str):
    """
    Menampilkan judul bagian di terminal agar proses eksekusi mudah dibaca.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def get_mode_value(series: pd.Series):
    """
    Mengambil nilai modus dari sebuah Series.
    Jika modus kosong, fungsi mengembalikan nilai pertama yang tersedia.
    """
    non_null_series = series.dropna()

    if non_null_series.empty:
        return None

    mode_values = non_null_series.mode()

    if not mode_values.empty:
        return mode_values.iloc[0]

    return non_null_series.iloc[0]


def clean_text_value(value):
    """
    Membersihkan nilai teks:
    - menghapus spasi di awal dan akhir,
    - menyeragamkan format menjadi title case.
    """
    if pd.isna(value):
        return value

    return str(value).strip().title()