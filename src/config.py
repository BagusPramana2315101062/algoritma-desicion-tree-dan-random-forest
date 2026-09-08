from pathlib import Path

# ============================================================
# KONFIGURASI PATH PROJECT
# ============================================================

# ROOT_DIR = folder utama project
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

VISUAL_DIR = OUTPUT_DIR / "visualisasi"
VISUAL_BEFORE_DIR = VISUAL_DIR / "01_distribusi_sebelum_preprocessing"
VISUAL_AFTER_DIR = VISUAL_DIR / "02_distribusi_sesudah_preprocessing"
VISUAL_AUDIT_DIR = VISUAL_DIR / "03_audit_preprocessing"
VISUAL_EVALUATION_DIR = VISUAL_DIR / "04_evaluasi_model"
VISUAL_FEATURE_IMPORTANCE_DIR = VISUAL_DIR / "05_feature_importance"

DATASET_PATH = DATA_DIR / "student_data_raw.xlsx"

# ============================================================
# KONFIGURASI DATASET
# ============================================================

ID_COLUMN = "ID_Mahasiswa"
TARGET_COLUMN = "Status_Kelulusan"

NUMERIC_FEATURES = [
    "IPK",
    "Kehadiran",
    "Jumlah_Organisasi",
    "Total_SKS",
    "Jam_Belajar",
]

CATEGORICAL_FEATURES = [
    "Penghasilan_Orang_Tua",
    "Kualitas_Internet",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ALL_COLUMNS = [
    ID_COLUMN,
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
]

# ============================================================
# KONFIGURASI MODEL
# ============================================================

TEST_SIZE = 0.2
RANDOM_STATE = 42

DECISION_TREE_MODEL_PATH = MODEL_DIR / "model_decision_tree.pkl"
RANDOM_FOREST_MODEL_PATH = MODEL_DIR / "model_random_forest.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

# ============================================================
# KONFIGURASI OUTPUT
# ============================================================

AUDIT_DATASET_PATH = OUTPUT_DIR / "audit_dataset_awal.csv"
DUPLICATE_AUDIT_PATH = OUTPUT_DIR / "audit_duplikat_penuh.csv"
CONFLICT_ID_AUDIT_PATH = OUTPUT_DIR / "audit_konflik_id.csv"
MISSING_VALUE_AUDIT_PATH = OUTPUT_DIR / "audit_missing_value.csv"
IQR_RESULT_PATH = OUTPUT_DIR / "hasil_iqr_outlier.csv"
FINAL_DATASET_PATH = OUTPUT_DIR / "dataset_setelah_preprocessing.csv"
EVALUATION_RESULT_PATH = OUTPUT_DIR / "hasil_evaluasi_model.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance_random_forest.csv"