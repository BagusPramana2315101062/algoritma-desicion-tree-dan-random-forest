import numpy as np
import pandas as pd

from config import (
    DATASET_PATH,
    ID_COLUMN,
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    ALL_COLUMNS,
)

from utils import get_mode_value, clean_text_value


# ============================================================
# LOAD DAN VALIDASI DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Membaca dataset Excel dari folder data.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di path: {DATASET_PATH}\n"
            "Pastikan file dataset_uts_dirty_preprocessing.xlsx berada di folder data."
        )

    dataframe = pd.read_excel(DATASET_PATH)
    validate_dataset_columns(dataframe)

    return dataframe


def validate_dataset_columns(dataframe: pd.DataFrame):
    """
    Memastikan seluruh kolom yang dibutuhkan tersedia dalam dataset.
    """
    missing_columns = [column for column in ALL_COLUMNS if column not in dataframe.columns]

    if missing_columns:
        raise ValueError(
            "Dataset tidak memiliki kolom wajib berikut: "
            + ", ".join(missing_columns)
        )


# ============================================================
# AUDIT AWAL DATASET
# ============================================================

def create_initial_dataset_audit(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat ringkasan audit awal dataset.
    """
    total_rows = len(dataframe)
    total_columns = dataframe.shape[1]
    unique_ids = dataframe[ID_COLUMN].nunique(dropna=True)
    full_duplicates = dataframe.duplicated().sum()

    duplicated_id_rows = dataframe[dataframe.duplicated(subset=[ID_COLUMN], keep=False)]
    duplicated_id_count = duplicated_id_rows[ID_COLUMN].nunique(dropna=True)
    duplicated_id_row_count = len(duplicated_id_rows)

    audit_data = [
        ["Jumlah baris awal", total_rows],
        ["Jumlah kolom", total_columns],
        ["Jumlah ID mahasiswa unik", unique_ids],
        ["Jumlah duplikat penuh", full_duplicates],
        ["Jumlah ID yang muncul lebih dari sekali", duplicated_id_count],
        ["Jumlah baris dengan ID berulang", duplicated_id_row_count],
    ]

    return pd.DataFrame(audit_data, columns=["Komponen", "Nilai"])


def create_missing_value_audit(dataframe_before: pd.DataFrame, dataframe_after: pd.DataFrame = None) -> pd.DataFrame:
    """
    Membuat audit missing value.
    Jika dataframe_after diberikan, audit akan membandingkan sebelum dan sesudah imputasi.
    """
    rows = []

    for column in ALL_COLUMNS:
        missing_before = dataframe_before[column].isna().sum()

        if dataframe_after is not None:
            missing_after = dataframe_after[column].isna().sum()
        else:
            missing_after = np.nan

        rows.append({
            "Kolom": column,
            "Missing_Sebelum": missing_before,
            "Missing_Sesudah": missing_after,
        })

    return pd.DataFrame(rows)


# ============================================================
# DUPLIKAT PENUH
# ============================================================

def remove_full_duplicates(dataframe: pd.DataFrame):
    """
    Menghapus duplikat penuh, yaitu baris yang seluruh nilainya identik.
    """
    rows_before = len(dataframe)
    duplicate_count = dataframe.duplicated().sum()

    cleaned_dataframe = dataframe.drop_duplicates().copy()

    rows_after = len(cleaned_dataframe)
    duplicate_after = cleaned_dataframe.duplicated().sum()

    audit = pd.DataFrame([
        {
            "Tahap": "Sebelum hapus duplikat penuh",
            "Jumlah_Baris": rows_before,
            "Jumlah_Duplikat_Penuh": duplicate_count,
            "Jumlah_ID_Unik": dataframe[ID_COLUMN].nunique(dropna=True),
        },
        {
            "Tahap": "Sesudah hapus duplikat penuh",
            "Jumlah_Baris": rows_after,
            "Jumlah_Duplikat_Penuh": duplicate_after,
            "Jumlah_ID_Unik": cleaned_dataframe[ID_COLUMN].nunique(dropna=True),
        },
    ])

    return cleaned_dataframe, audit


# ============================================================
# KONFLIK ID MAHASISWA
# ============================================================

def get_conflict_id_detail(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Menampilkan ID mahasiswa yang muncul lebih dari satu kali.
    Kolom yang berbeda dicatat agar konflik dapat dianalisis.
    """
    duplicated_id_rows = dataframe[dataframe.duplicated(subset=[ID_COLUMN], keep=False)].copy()

    if duplicated_id_rows.empty:
        return pd.DataFrame(columns=[
            "ID_Mahasiswa",
            "Jumlah_Record",
            "Kolom_Yang_Berbeda",
            "Target",
        ])

    conflict_rows = []

    for student_id, group in duplicated_id_rows.groupby(ID_COLUMN):
        different_columns = []

        for column in dataframe.columns:
            if column == ID_COLUMN:
                continue

            unique_values = group[column].drop_duplicates()

            if len(unique_values) > 1:
                different_columns.append(column)

        target_values = group[TARGET_COLUMN].dropna().drop_duplicates().astype(str).tolist()

        conflict_rows.append({
            "ID_Mahasiswa": student_id,
            "Jumlah_Record": len(group),
            "Kolom_Yang_Berbeda": ", ".join(different_columns) if different_columns else "-",
            "Target": ", ".join(target_values) if target_values else "-",
        })

    return pd.DataFrame(conflict_rows)


def create_conflict_id_audit(dataframe_before: pd.DataFrame, dataframe_after: pd.DataFrame = None) -> pd.DataFrame:
    """
    Membuat audit jumlah konflik ID sebelum dan sesudah agregasi.
    """
    duplicated_before = dataframe_before[dataframe_before.duplicated(subset=[ID_COLUMN], keep=False)]

    before_conflict_ids = duplicated_before[ID_COLUMN].nunique(dropna=True)
    before_conflict_rows = len(duplicated_before)

    if dataframe_after is not None:
        duplicated_after = dataframe_after[dataframe_after.duplicated(subset=[ID_COLUMN], keep=False)]
        after_conflict_ids = duplicated_after[ID_COLUMN].nunique(dropna=True)
        after_conflict_rows = len(duplicated_after)
        after_rows = len(dataframe_after)
        after_unique_ids = dataframe_after[ID_COLUMN].nunique(dropna=True)
    else:
        after_conflict_ids = np.nan
        after_conflict_rows = np.nan
        after_rows = np.nan
        after_unique_ids = np.nan

    audit = pd.DataFrame([
        {
            "Tahap": "Sebelum agregasi konflik ID",
            "Jumlah_Baris": len(dataframe_before),
            "Jumlah_ID_Unik": dataframe_before[ID_COLUMN].nunique(dropna=True),
            "Jumlah_ID_Bermasalah": before_conflict_ids,
            "Jumlah_Baris_ID_Bermasalah": before_conflict_rows,
        },
        {
            "Tahap": "Sesudah agregasi konflik ID",
            "Jumlah_Baris": after_rows,
            "Jumlah_ID_Unik": after_unique_ids,
            "Jumlah_ID_Bermasalah": after_conflict_ids,
            "Jumlah_Baris_ID_Bermasalah": after_conflict_rows,
        },
    ])

    return audit


def aggregate_conflict_ids(dataframe: pd.DataFrame):
    """
    Menangani konflik ID dengan agregasi per ID mahasiswa.

    Aturan agregasi:
    - Kolom numerik menggunakan median.
    - Kolom kategorikal menggunakan modus.
    - Target menggunakan modus.
    """
    dataframe = dataframe.copy()

    aggregation_rules = {}

    for column in NUMERIC_FEATURES:
        aggregation_rules[column] = "median"

    for column in CATEGORICAL_FEATURES:
        aggregation_rules[column] = get_mode_value

    aggregation_rules[TARGET_COLUMN] = get_mode_value

    aggregated_dataframe = (
        dataframe
        .groupby(ID_COLUMN, as_index=False)
        .agg(aggregation_rules)
    )

    # Susun ulang kolom agar sama seperti dataset awal
    aggregated_dataframe = aggregated_dataframe[ALL_COLUMNS]

    audit = create_conflict_id_audit(dataframe, aggregated_dataframe)
    conflict_detail_before = get_conflict_id_detail(dataframe)
    conflict_detail_after = get_conflict_id_detail(aggregated_dataframe)

    return aggregated_dataframe, audit, conflict_detail_before, conflict_detail_after


# ============================================================
# STANDARDISASI KATEGORI
# ============================================================

def create_category_count_table(dataframe: pd.DataFrame, stage_name: str) -> pd.DataFrame:
    """
    Membuat tabel jumlah kategori untuk fitur kategorikal dan target.
    """
    rows = []

    category_columns = CATEGORICAL_FEATURES + [TARGET_COLUMN]

    for column in category_columns:
        value_counts = dataframe[column].value_counts(dropna=False)

        for value, count in value_counts.items():
            rows.append({
                "Tahap": stage_name,
                "Kolom": column,
                "Kategori": "Missing" if pd.isna(value) else str(value),
                "Jumlah": int(count),
            })

    return pd.DataFrame(rows)


def standardize_categories(dataframe: pd.DataFrame):
    """
    Menstandarkan penulisan kategori:
    - spasi awal/akhir dihapus,
    - format huruf dibuat Title Case.
    """
    dataframe = dataframe.copy()

    category_before = create_category_count_table(dataframe, "Sebelum standardisasi")

    columns_to_standardize = CATEGORICAL_FEATURES + [TARGET_COLUMN]

    for column in columns_to_standardize:
        dataframe[column] = dataframe[column].apply(clean_text_value)

    category_after = create_category_count_table(dataframe, "Sesudah standardisasi")

    category_audit = pd.concat([category_before, category_after], ignore_index=True)

    return dataframe, category_audit


# ============================================================
# IMPUTASI MISSING VALUE
# ============================================================

def impute_missing_values(dataframe: pd.DataFrame):
    """
    Menangani missing value:
    - Fitur numerik menggunakan median.
    - Fitur kategorikal menggunakan modus.
    - Target kosong, jika ada, dihapus karena target tidak sebaiknya diimputasi.
    """
    dataframe = dataframe.copy()

    missing_before_dataframe = dataframe.copy()

    # Jika target kosong, hapus baris tersebut.
    dataframe = dataframe.dropna(subset=[TARGET_COLUMN]).copy()

    imputation_summary = []

    for column in NUMERIC_FEATURES:
        median_value = dataframe[column].median()
        missing_count = dataframe[column].isna().sum()

        dataframe[column] = dataframe[column].fillna(median_value)

        imputation_summary.append({
            "Kolom": column,
            "Jenis": "Numerik",
            "Metode_Imputasi": "Median",
            "Nilai_Imputasi": median_value,
            "Jumlah_Missing_Diisi": int(missing_count),
        })

    for column in CATEGORICAL_FEATURES:
        mode_value = get_mode_value(dataframe[column])
        missing_count = dataframe[column].isna().sum()

        dataframe[column] = dataframe[column].fillna(mode_value)

        imputation_summary.append({
            "Kolom": column,
            "Jenis": "Kategorikal",
            "Metode_Imputasi": "Modus",
            "Nilai_Imputasi": mode_value,
            "Jumlah_Missing_Diisi": int(missing_count),
        })

    missing_audit = create_missing_value_audit(missing_before_dataframe, dataframe)
    imputation_summary = pd.DataFrame(imputation_summary)

    return dataframe, missing_audit, imputation_summary


# ============================================================
# OUTLIER HANDLING DENGAN IQR CAPPING
# ============================================================

def calculate_iqr_bounds(series: pd.Series):
    """
    Menghitung batas bawah dan batas atas IQR.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return q1, q3, iqr, lower_bound, upper_bound


def count_outliers(series: pd.Series, lower_bound: float, upper_bound: float) -> int:
    """
    Menghitung jumlah outlier berdasarkan batas IQR.
    """
    return int(((series < lower_bound) | (series > upper_bound)).sum())


def apply_iqr_capping(dataframe: pd.DataFrame):
    """
    Mendeteksi dan menangani outlier pada fitur numerik menggunakan IQR capping.

    Nilai yang lebih kecil dari batas bawah diganti menjadi batas bawah.
    Nilai yang lebih besar dari batas atas diganti menjadi batas atas.
    """
    dataframe = dataframe.copy()
    iqr_rows = []

    for column in NUMERIC_FEATURES:
        q1, q3, iqr, lower_bound, upper_bound = calculate_iqr_bounds(dataframe[column])

        outlier_before = count_outliers(dataframe[column], lower_bound, upper_bound)

        dataframe[column] = dataframe[column].clip(lower=lower_bound, upper=upper_bound)

        outlier_after = count_outliers(dataframe[column], lower_bound, upper_bound)

        iqr_rows.append({
            "Fitur": column,
            "Q1": round(q1, 3),
            "Q3": round(q3, 3),
            "IQR": round(iqr, 3),
            "Batas_Bawah": round(lower_bound, 3),
            "Batas_Atas": round(upper_bound, 3),
            "Outlier_Sebelum": outlier_before,
            "Outlier_Sesudah": outlier_after,
            "Metode": "Capping",
        })

    iqr_result = pd.DataFrame(iqr_rows)

    return dataframe, iqr_result


# ============================================================
# PEMISAHAN FITUR DAN TARGET
# ============================================================

def split_features_target(dataframe: pd.DataFrame):
    """
    Memisahkan dataset menjadi fitur X dan target y.
    """
    X = dataframe[FEATURE_COLUMNS].copy()
    y = dataframe[TARGET_COLUMN].copy()

    return X, y