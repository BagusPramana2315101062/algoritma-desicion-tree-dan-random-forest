import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =====================================================
# 1. KONFIGURASI PATH
# =====================================================

DATA_PATH = "data/dataset_uts_dirty_preprocessing.xlsx"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
VISUAL_DIR = os.path.join(OUTPUT_DIR, "visualisasi")

# Fallback agar kode tetap bisa dijalankan di environment ChatGPT/sandbox.
# Jika dijalankan di laptop, cukup gunakan struktur folder data/ seperti biasa.
if not os.path.exists(DATA_PATH):
    alternatif_path = "/mnt/data/dataset_uts_dirty_preprocessing.xlsx"
    if os.path.exists(alternatif_path):
        DATA_PATH = alternatif_path

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)


# =====================================================
# 2. DAFTAR FITUR
# =====================================================

numeric_features = [
    "IPK",
    "Kehadiran",
    "Jumlah_Organisasi",
    "Total_SKS",
    "Jam_Belajar"
]

categorical_features = [
    "Penghasilan_Orang_Tua",
    "Kualitas_Internet"
]

target_col = "Status_Kelulusan"

kolom_kategori = [
    "Penghasilan_Orang_Tua",
    "Kualitas_Internet",
    "Status_Kelulusan"
]


# =====================================================
# 3. FUNGSI BANTU VISUALISASI
# =====================================================

def simpan_gambar(nama_file):
    """Menyimpan gambar ke folder visualisasi dengan resolusi tinggi."""
    path = os.path.join(VISUAL_DIR, nama_file)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualisasi disimpan: {path}")


def tambah_label_bar(ax, persen=False):
    """Menambahkan angka di atas bar chart."""
    for bar in ax.patches:
        tinggi = bar.get_height()
        if pd.isna(tinggi):
            continue
        label = f"{tinggi:.2f}%" if persen else f"{int(tinggi)}"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, tinggi),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points"
        )


def hitung_missing_lengkap(dataframe):
    """
    Menghitung missing value, termasuk nilai kosong yang berbentuk string
    seperti '', ' ', 'None', dan 'Nan'.
    """
    df_temp = dataframe.copy()
    df_temp = df_temp.replace(["", " ", "None", "none", "NONE", "Nan", "nan", "NaN"], np.nan)
    return df_temp.isna().sum()


def standardisasi_kolom_kategori(dataframe):
    """Menyeragamkan kapitalisasi kategori agar visualisasi lebih rapi."""
    df_temp = dataframe.copy()
    for col in kolom_kategori:
        if col in df_temp.columns:
            df_temp[col] = df_temp[col].astype(str).str.strip().str.title()
    df_temp = df_temp.replace(["Nan", "None", "", " "], np.nan)
    return df_temp


# =====================================================
# 4. VISUALISASI DATA SEBELUM PREPROCESSING
# =====================================================

def visualisasi_sebelum_preprocessing(df_raw):
    """Membuat visualisasi kondisi data mentah sebelum preprocessing."""
    print("\nMembuat visualisasi SEBELUM preprocessing...")

    # 1) Missing value sebelum preprocessing
    missing_counts = hitung_missing_lengkap(df_raw)
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if len(missing_counts) > 0:
        fig, ax = plt.subplots(figsize=(9, 5))
        missing_counts.plot(kind="bar", ax=ax)
        ax.set_title("Missing Value Sebelum Preprocessing")
        ax.set_xlabel("Kolom")
        ax.set_ylabel("Jumlah Missing Value")
        ax.tick_params(axis="x", rotation=30)
        tambah_label_bar(ax)
        simpan_gambar("01_missing_value_sebelum_preprocessing.png")

    # 2) Distribusi target sebelum preprocessing
    if target_col in df_raw.columns:
        target_raw = df_raw[target_col].astype(str).str.strip().str.title().fillna("Missing")
        target_counts = target_raw.value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        target_counts.plot(kind="bar", ax=ax)
        ax.set_title("Distribusi Target Sebelum Preprocessing")
        ax.set_xlabel("Status Kelulusan")
        ax.set_ylabel("Jumlah Data")
        ax.tick_params(axis="x", rotation=0)
        tambah_label_bar(ax)
        simpan_gambar("02_distribusi_target_sebelum_preprocessing.png")

    # 3) Inkonsistensi kategori sebelum standardisasi
    for col in categorical_features:
        if col in df_raw.columns:
            kategori_counts = df_raw[col].astype(str).fillna("Missing").value_counts(dropna=False)

            fig, ax = plt.subplots(figsize=(9, 5))
            kategori_counts.plot(kind="bar", ax=ax)
            ax.set_title(f"Kategori {col} Sebelum Standardisasi")
            ax.set_xlabel("Kategori")
            ax.set_ylabel("Jumlah Data")
            ax.tick_params(axis="x", rotation=30)
            tambah_label_bar(ax)
            simpan_gambar(f"03_kategori_{col.lower()}_sebelum_standardisasi.png")

    # 4) Boxplot fitur numerik sebelum preprocessing untuk melihat outlier
    numeric_existing = [col for col in numeric_features if col in df_raw.columns]
    df_num_raw = df_raw[numeric_existing].copy()
    for col in numeric_existing:
        df_num_raw[col] = pd.to_numeric(df_num_raw[col], errors="coerce")

    if len(numeric_existing) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        df_num_raw.boxplot(ax=ax)
        ax.set_title("Boxplot Fitur Numerik Sebelum Preprocessing")
        ax.set_xlabel("Fitur Numerik")
        ax.set_ylabel("Nilai")
        ax.tick_params(axis="x", rotation=30)
        simpan_gambar("04_boxplot_numerik_sebelum_preprocessing.png")

    # 5) Scatter IPK vs Jam_Belajar sebelum preprocessing
    if {"IPK", "Jam_Belajar", target_col}.issubset(df_raw.columns):
        df_scatter = df_raw[["IPK", "Jam_Belajar", target_col]].copy()
        df_scatter["IPK"] = pd.to_numeric(df_scatter["IPK"], errors="coerce")
        df_scatter["Jam_Belajar"] = pd.to_numeric(df_scatter["Jam_Belajar"], errors="coerce")
        df_scatter[target_col] = df_scatter[target_col].astype(str).str.strip().str.title()
        df_scatter = df_scatter.dropna(subset=["IPK", "Jam_Belajar", target_col])

        fig, ax = plt.subplots(figsize=(8, 6))
        for label, group in df_scatter.groupby(target_col):
            ax.scatter(group["IPK"], group["Jam_Belajar"], label=label, alpha=0.7)
        ax.set_title("Scatter Plot IPK vs Jam_Belajar Sebelum Preprocessing")
        ax.set_xlabel("IPK")
        ax.set_ylabel("Jam_Belajar")
        ax.legend(title="Status Kelulusan")
        simpan_gambar("05_scatter_ipk_jambelajar_sebelum_preprocessing.png")


# =====================================================
# 5. VISUALISASI DATA SETELAH PREPROCESSING
# =====================================================

def buat_dataframe_setelah_preprocessing(df_clean):
    """
    Membuat salinan dataframe setelah preprocessing untuk visualisasi.
    Data dibuat tetap mudah dibaca, sehingga kategori tidak diubah ke angka.
    """
    df_after = df_clean.copy()

    # Pastikan tipe numerik benar
    for col in numeric_features:
        if col in df_after.columns:
            df_after[col] = pd.to_numeric(df_after[col], errors="coerce")

    # Imputasi numerik dengan median
    for col in numeric_features:
        if col in df_after.columns:
            median_value = df_after[col].median()
            df_after[col] = df_after[col].fillna(median_value)

    # Imputasi kategorikal dengan modus
    for col in categorical_features:
        if col in df_after.columns:
            mode_value = df_after[col].mode(dropna=True)
            if len(mode_value) > 0:
                df_after[col] = df_after[col].fillna(mode_value.iloc[0])

    return df_after


def visualisasi_setelah_preprocessing(df_after, jumlah_awal, jumlah_setelah_duplikat, jumlah_akhir):
    """Membuat visualisasi kondisi data setelah preprocessing."""
    print("\nMembuat visualisasi SETELAH preprocessing...")

    # 1) Ringkasan jumlah data setelah tahapan cleaning
    ringkasan_data = pd.Series({
        "Data awal": jumlah_awal,
        "Setelah hapus duplikat": jumlah_setelah_duplikat,
        "Setelah hapus ID konflik": jumlah_akhir
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    ringkasan_data.plot(kind="bar", ax=ax)
    ax.set_title("Perubahan Jumlah Data Setelah Cleaning")
    ax.set_xlabel("Tahapan")
    ax.set_ylabel("Jumlah Data")
    ax.tick_params(axis="x", rotation=15)
    tambah_label_bar(ax)
    simpan_gambar("06_perubahan_jumlah_data_setelah_cleaning.png")

    # 2) Missing value setelah imputasi
    missing_after = hitung_missing_lengkap(df_after)
    missing_after = missing_after[missing_after.index.isin(numeric_features + categorical_features)]

    fig, ax = plt.subplots(figsize=(9, 5))
    missing_after.plot(kind="bar", ax=ax)
    ax.set_title("Missing Value Setelah Preprocessing")
    ax.set_xlabel("Kolom")
    ax.set_ylabel("Jumlah Missing Value")
    ax.tick_params(axis="x", rotation=30)
    tambah_label_bar(ax)
    simpan_gambar("07_missing_value_setelah_preprocessing.png")

    # 3) Distribusi target setelah cleaning
    if target_col in df_after.columns:
        target_counts = df_after[target_col].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        target_counts.plot(kind="bar", ax=ax)
        ax.set_title("Distribusi Target Setelah Preprocessing")
        ax.set_xlabel("Status Kelulusan")
        ax.set_ylabel("Jumlah Data")
        ax.tick_params(axis="x", rotation=0)
        tambah_label_bar(ax)
        simpan_gambar("08_distribusi_target_setelah_preprocessing.png")

    # 4) Kategori setelah standardisasi dan imputasi
    for col in categorical_features:
        if col in df_after.columns:
            kategori_counts = df_after[col].value_counts(dropna=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            kategori_counts.plot(kind="bar", ax=ax)
            ax.set_title(f"Kategori {col} Setelah Standardisasi")
            ax.set_xlabel("Kategori")
            ax.set_ylabel("Jumlah Data")
            ax.tick_params(axis="x", rotation=20)
            tambah_label_bar(ax)
            simpan_gambar(f"09_kategori_{col.lower()}_setelah_standardisasi.png")

    # 5) Boxplot fitur numerik setelah imputasi
    numeric_existing = [col for col in numeric_features if col in df_after.columns]
    if len(numeric_existing) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        df_after[numeric_existing].boxplot(ax=ax)
        ax.set_title("Boxplot Fitur Numerik Setelah Preprocessing")
        ax.set_xlabel("Fitur Numerik")
        ax.set_ylabel("Nilai")
        ax.tick_params(axis="x", rotation=30)
        simpan_gambar("10_boxplot_numerik_setelah_preprocessing.png")

    # 6) Scatter IPK vs Jam_Belajar setelah preprocessing
    if {"IPK", "Jam_Belajar", target_col}.issubset(df_after.columns):
        fig, ax = plt.subplots(figsize=(8, 6))
        for label, group in df_after.groupby(target_col):
            ax.scatter(group["IPK"], group["Jam_Belajar"], label=label, alpha=0.7)
        ax.set_title("Scatter Plot IPK vs Jam_Belajar Setelah Preprocessing")
        ax.set_xlabel("IPK")
        ax.set_ylabel("Jam_Belajar")
        ax.legend(title="Status Kelulusan")
        simpan_gambar("11_scatter_ipk_jambelajar_setelah_preprocessing.png")

    # 7) Risiko Tidak Tepat Waktu berdasarkan kelompok IPK
    if {"IPK", target_col}.issubset(df_after.columns):
        df_risiko_ipk = df_after.copy()
        df_risiko_ipk["Kelompok_IPK"] = pd.cut(
            df_risiko_ipk["IPK"],
            bins=[0, 2.5, 3.0, 3.5, 4.1],
            labels=["< 2,50", "2,50 - 2,99", "3,00 - 3,49", "≥ 3,50"],
            right=False
        )
        risiko_ipk = (
            df_risiko_ipk.groupby("Kelompok_IPK", observed=False)[target_col]
            .apply(lambda x: (x == "Tidak Tepat Waktu").mean() * 100)
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        risiko_ipk.plot(kind="bar", ax=ax)
        ax.set_title("Persentase Tidak Tepat Waktu Berdasarkan Kelompok IPK")
        ax.set_xlabel("Kelompok IPK")
        ax.set_ylabel("Persentase Tidak Tepat Waktu (%)")
        ax.tick_params(axis="x", rotation=15)
        tambah_label_bar(ax, persen=True)
        simpan_gambar("12_risiko_tidak_tepat_waktu_berdasarkan_ipk.png")

    # 8) Risiko Tidak Tepat Waktu berdasarkan kelompok Jam_Belajar
    if {"Jam_Belajar", target_col}.issubset(df_after.columns):
        df_risiko_jam = df_after.copy()
        df_risiko_jam["Kelompok_Jam_Belajar"] = pd.cut(
            df_risiko_jam["Jam_Belajar"],
            bins=[-1, 10, 20, 30, 40, float("inf")],
            labels=["< 10", "10 - 19", "20 - 29", "30 - 39", "≥ 40"],
            right=False
        )
        risiko_jam = (
            df_risiko_jam.groupby("Kelompok_Jam_Belajar", observed=False)[target_col]
            .apply(lambda x: (x == "Tidak Tepat Waktu").mean() * 100)
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        risiko_jam.plot(kind="bar", ax=ax)
        ax.set_title("Persentase Tidak Tepat Waktu Berdasarkan Kelompok Jam_Belajar")
        ax.set_xlabel("Kelompok Jam_Belajar")
        ax.set_ylabel("Persentase Tidak Tepat Waktu (%)")
        ax.tick_params(axis="x", rotation=15)
        tambah_label_bar(ax, persen=True)
        simpan_gambar("13_risiko_tidak_tepat_waktu_berdasarkan_jam_belajar.png")


def visualisasi_hasil_model(hasil_evaluasi_final, feature_importance):
    """Membuat visualisasi tambahan untuk hasil model."""
    print("\nMembuat visualisasi hasil model...")

    # 1) Perbandingan metrik evaluasi model
    metrik_cols = ["Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro"]
    df_plot = hasil_evaluasi_final.set_index("Model")[metrik_cols] * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    df_plot.plot(kind="bar", ax=ax)
    ax.set_title("Perbandingan Evaluasi Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Nilai Metrik (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Metrik")
    simpan_gambar("14_perbandingan_evaluasi_model.png")

    # 2) Feature importance Random Forest terbaik
    fig, ax = plt.subplots(figsize=(9, 5))
    feature_importance.sort_values("Importance", ascending=True).plot(
        kind="barh",
        x="Fitur",
        y="Importance",
        ax=ax,
        legend=False
    )
    ax.set_title("Feature Importance Random Forest Terbaik")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Fitur")
    simpan_gambar("15_feature_importance_random_forest.png")


# =====================================================
# 6. MEMBACA DATASET
# =====================================================

print("Membaca dataset...")

df = pd.read_excel(DATA_PATH)
df_raw = df.copy()

print("\nInformasi awal dataset:")
print("Jumlah baris dan kolom:", df.shape)
print("Daftar kolom:", list(df.columns))

# Visualisasi kondisi data mentah sebelum preprocessing
visualisasi_sebelum_preprocessing(df_raw)


# =====================================================
# 7. MENGHAPUS DUPLIKAT PENUH
# =====================================================

jumlah_awal = len(df)
df = df.drop_duplicates()
jumlah_setelah_duplikat = len(df)

print("\nJumlah data awal:", jumlah_awal)
print("Jumlah data setelah hapus duplikat penuh:", jumlah_setelah_duplikat)
print("Jumlah duplikat penuh yang dihapus:", jumlah_awal - jumlah_setelah_duplikat)


# =====================================================
# 8. MENANGANI ID KONFLIK
# =====================================================

if "ID_Mahasiswa" in df.columns:
    duplicate_ids = df["ID_Mahasiswa"].value_counts()
    duplicate_ids = duplicate_ids[duplicate_ids > 1].index

    print("\nJumlah ID yang muncul lebih dari satu kali:", len(duplicate_ids))

    if len(duplicate_ids) > 0:
        print("ID konflik/berulang yang dikeluarkan dari data model:")
        print(list(duplicate_ids))

        df = df[~df["ID_Mahasiswa"].isin(duplicate_ids)]

print("Jumlah data setelah penanganan ID konflik:", len(df))
jumlah_akhir_cleaning = len(df)


# =====================================================
# 9. STANDARDISASI KATEGORI
# =====================================================

df = standardisasi_kolom_kategori(df)

print("\nCek kategori setelah standardisasi:")
for col in kolom_kategori:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False))

# Buat dataframe visual setelah preprocessing sederhana: cleaning + standardisasi + imputasi.
df_after_visual = buat_dataframe_setelah_preprocessing(df)
visualisasi_setelah_preprocessing(
    df_after_visual,
    jumlah_awal=jumlah_awal,
    jumlah_setelah_duplikat=jumlah_setelah_duplikat,
    jumlah_akhir=jumlah_akhir_cleaning
)


# =====================================================
# 10. MEMISAHKAN FITUR DAN TARGET
# =====================================================

if target_col not in df.columns:
    raise ValueError(f"Kolom target '{target_col}' tidak ditemukan dalam dataset.")

X = df.drop(columns=["ID_Mahasiswa", target_col], errors="ignore")
y = df[target_col]

# Encoding target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nLabel target:")
for index, label in enumerate(label_encoder.classes_):
    print(f"{index} = {label}")

# Pastikan kolom numerik benar-benar bertipe numerik
for col in numeric_features:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")


# =====================================================
# 11. MEMBUAT PREPROCESSING PIPELINE UNTUK MODEL
# =====================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=[
            ["Rendah", "Menengah", "Tinggi"],
            ["Buruk", "Sedang", "Baik"]
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# =====================================================
# 12. SPLIT DATA LATIH DAN DATA UJI
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nJumlah data latih:", len(X_train))
print("Jumlah data uji:", len(X_test))


# =====================================================
# 13. MEMBUAT MODEL BASELINE
# =====================================================

dt_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ))
])

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        class_weight="balanced",
        random_state=42
    ))
])


# =====================================================
# 14. FUNGSI EVALUASI MODEL
# =====================================================

def evaluasi_model(nama_model, model, X_test, y_test):
    print(f"\n==============================")
    print(f"EVALUASI MODEL: {nama_model}")
    print(f"==============================")

    y_pred = model.predict(X_test)

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )

    print(report_text)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    hasil_ringkas = {
        "Model": nama_model,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision_Macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall_Macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1_Macro": f1_score(y_test, y_pred, average="macro", zero_division=0)
    }

    # Simpan classification report ke txt
    report_path = os.path.join(
        OUTPUT_DIR,
        f"classification_report_{nama_model.lower().replace(' ', '_')}.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    # Simpan confusion matrix sebagai gambar
    plt.figure(figsize=(6, 4))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {nama_model}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, rotation=30)
    plt.yticks(range(len(label_encoder.classes_)), label_encoder.classes_)

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    cm_path = os.path.join(
        OUTPUT_DIR,
        f"confusion_matrix_{nama_model.lower().replace(' ', '_')}.png"
    )
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    return hasil_ringkas


# =====================================================
# 15. TRAINING DAN EVALUASI BASELINE
# =====================================================

print("\nMelatih model Decision Tree...")
dt_model.fit(X_train, y_train)

print("Melatih model Random Forest...")
rf_model.fit(X_train, y_train)

hasil_dt = evaluasi_model("Decision Tree", dt_model, X_test, y_test)
hasil_rf = evaluasi_model("Random Forest", rf_model, X_test, y_test)

hasil_evaluasi = pd.DataFrame([hasil_dt, hasil_rf])
hasil_evaluasi.to_csv(
    os.path.join(OUTPUT_DIR, "hasil_evaluasi_baseline.csv"),
    index=False
)

print("\nRingkasan evaluasi baseline:")
print(hasil_evaluasi)


# =====================================================
# 16. HYPERPARAMETER TUNING
# =====================================================

print("\nMelakukan tuning Decision Tree...")

param_dt = {
    "classifier__max_depth": [3, 5, 7, 10, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}

grid_dt = GridSearchCV(
    dt_model,
    param_grid=param_dt,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

grid_dt.fit(X_train, y_train)

print("Parameter terbaik Decision Tree:")
print(grid_dt.best_params_)


print("\nMelakukan tuning Random Forest...")

param_rf = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 7, 10, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}

grid_rf = GridSearchCV(
    rf_model,
    param_grid=param_rf,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)

print("Parameter terbaik Random Forest:")
print(grid_rf.best_params_)


# =====================================================
# 17. EVALUASI MODEL HASIL TUNING
# =====================================================

best_dt_model = grid_dt.best_estimator_
best_rf_model = grid_rf.best_estimator_

hasil_dt_tuned = evaluasi_model("Decision Tree Tuned", best_dt_model, X_test, y_test)
hasil_rf_tuned = evaluasi_model("Random Forest Tuned", best_rf_model, X_test, y_test)

hasil_evaluasi_final = pd.DataFrame([
    hasil_dt,
    hasil_rf,
    hasil_dt_tuned,
    hasil_rf_tuned
])

hasil_evaluasi_final.to_csv(
    os.path.join(OUTPUT_DIR, "hasil_evaluasi_final.csv"),
    index=False
)

print("\nRingkasan evaluasi final:")
print(hasil_evaluasi_final)


# =====================================================
# 18. FEATURE IMPORTANCE RANDOM FOREST TERBAIK
# =====================================================

fitur_model = numeric_features + categorical_features

rf_classifier = best_rf_model.named_steps["classifier"]
feature_importance = pd.DataFrame({
    "Fitur": fitur_model,
    "Importance": rf_classifier.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importance.to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance_random_forest.csv"),
    index=False
)

print("\nFeature Importance Random Forest:")
print(feature_importance)

# Visualisasi hasil model dan feature importance
visualisasi_hasil_model(hasil_evaluasi_final, feature_importance)


# =====================================================
# 19. MENYIMPAN MODEL
# =====================================================

joblib.dump(best_dt_model, os.path.join(MODEL_DIR, "model_decision_tree.pkl"))
joblib.dump(best_rf_model, os.path.join(MODEL_DIR, "model_random_forest.pkl"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("\nModel berhasil disimpan di folder models/")
print("Hasil evaluasi berhasil disimpan di folder outputs/")
print(f"Visualisasi berhasil disimpan di folder {VISUAL_DIR}/")
print("\nProses training selesai.")
