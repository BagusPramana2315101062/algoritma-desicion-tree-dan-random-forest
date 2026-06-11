import matplotlib.pyplot as plt
import pandas as pd

from config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    VISUAL_BEFORE_DIR,
    VISUAL_AFTER_DIR,
    VISUAL_AUDIT_DIR,
    VISUAL_EVALUATION_DIR,
    VISUAL_FEATURE_IMPORTANCE_DIR,
)


# ============================================================
# FUNGSI DASAR VISUALISASI
# ============================================================

def save_current_figure(output_path):
    """
    Menyimpan figure matplotlib ke file PNG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def format_filename(column_name: str) -> str:
    """
    Mengubah nama kolom menjadi format nama file yang konsisten.
    """
    return column_name.lower().replace(" ", "_")


def plot_numeric_distribution(dataframe: pd.DataFrame, column: str, output_dir, title_prefix: str):
    """
    Membuat histogram untuk fitur numerik.
    """
    plt.figure(figsize=(8, 5))

    series = dataframe[column].dropna()

    plt.hist(series, bins=20, edgecolor="black")
    plt.title(f"{title_prefix} - Distribusi {column}")
    plt.xlabel(column)
    plt.ylabel("Frekuensi")
    plt.grid(axis="y", alpha=0.3)

    output_path = output_dir / f"distribusi_{format_filename(column)}.png"
    save_current_figure(output_path)


def plot_categorical_distribution(dataframe: pd.DataFrame, column: str, output_dir, title_prefix: str):
    """
    Membuat bar chart untuk fitur kategorikal dan target.
    """
    plt.figure(figsize=(8, 5))

    counts = dataframe[column].fillna("Missing").astype(str).value_counts()

    plt.bar(counts.index, counts.values)
    plt.title(f"{title_prefix} - Distribusi {column}")
    plt.xlabel(column)
    plt.ylabel("Jumlah Data")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    output_path = output_dir / f"distribusi_{format_filename(column)}.png"
    save_current_figure(output_path)


def visualize_feature_distributions(dataframe: pd.DataFrame, output_dir, title_prefix: str):
    """
    Membuat visualisasi distribusi seluruh fitur dan target.
    - Fitur numerik divisualisasikan dengan histogram.
    - Fitur kategorikal dan target divisualisasikan dengan bar chart.
    """
    for column in NUMERIC_FEATURES:
        plot_numeric_distribution(dataframe, column, output_dir, title_prefix)

    for column in CATEGORICAL_FEATURES:
        plot_categorical_distribution(dataframe, column, output_dir, title_prefix)

    plot_categorical_distribution(dataframe, TARGET_COLUMN, output_dir, title_prefix)


def visualize_distributions_before_preprocessing(dataframe: pd.DataFrame):
    """
    Visualisasi distribusi seluruh fitur sebelum preprocessing.
    """
    visualize_feature_distributions(
        dataframe=dataframe,
        output_dir=VISUAL_BEFORE_DIR,
        title_prefix="Sebelum Preprocessing",
    )


def visualize_distributions_after_preprocessing(dataframe: pd.DataFrame):
    """
    Visualisasi distribusi seluruh fitur sesudah preprocessing.
    Visualisasi ini dibuat sebelum encoding kategori agar label kategori masih terbaca.
    """
    visualize_feature_distributions(
        dataframe=dataframe,
        output_dir=VISUAL_AFTER_DIR,
        title_prefix="Sesudah Preprocessing",
    )


# ============================================================
# VISUALISASI AUDIT PREPROCESSING
# ============================================================

def visualize_duplicate_audit(duplicate_audit: pd.DataFrame):
    """
    Membuat visualisasi audit duplikat penuh.
    """
    plt.figure(figsize=(8, 5))

    plt.bar(
        duplicate_audit["Tahap"],
        duplicate_audit["Jumlah_Duplikat_Penuh"],
    )

    plt.title("Audit Duplikat Penuh")
    plt.xlabel("Tahap")
    plt.ylabel("Jumlah Duplikat Penuh")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)

    output_path = VISUAL_AUDIT_DIR / "audit_duplikat_penuh.png"
    save_current_figure(output_path)


def visualize_row_count_audit(duplicate_audit: pd.DataFrame, conflict_audit: pd.DataFrame):
    """
    Membuat visualisasi perubahan jumlah baris:
    - Data awal
    - Setelah hapus duplikat penuh
    - Setelah agregasi konflik ID
    """
    rows = []

    if not duplicate_audit.empty:
        rows.append({
            "Tahap": "Data awal",
            "Jumlah_Baris": duplicate_audit.iloc[0]["Jumlah_Baris"],
        })
        rows.append({
            "Tahap": "Setelah hapus duplikat",
            "Jumlah_Baris": duplicate_audit.iloc[1]["Jumlah_Baris"],
        })

    if not conflict_audit.empty:
        rows.append({
            "Tahap": "Setelah agregasi ID",
            "Jumlah_Baris": conflict_audit.iloc[1]["Jumlah_Baris"],
        })

    audit_df = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    plt.bar(audit_df["Tahap"], audit_df["Jumlah_Baris"])

    plt.title("Audit Jumlah Baris Data")
    plt.xlabel("Tahap")
    plt.ylabel("Jumlah Baris")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)

    output_path = VISUAL_AUDIT_DIR / "audit_jumlah_baris.png"
    save_current_figure(output_path)


def visualize_conflict_id_audit(conflict_audit: pd.DataFrame):
    """
    Membuat visualisasi audit konflik ID sebelum dan sesudah agregasi.
    """
    plt.figure(figsize=(8, 5))

    plt.bar(
        conflict_audit["Tahap"],
        conflict_audit["Jumlah_ID_Bermasalah"],
    )

    plt.title("Audit Konflik ID Mahasiswa")
    plt.xlabel("Tahap")
    plt.ylabel("Jumlah ID Bermasalah")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)

    output_path = VISUAL_AUDIT_DIR / "audit_konflik_id.png"
    save_current_figure(output_path)


def visualize_missing_value_audit(missing_audit: pd.DataFrame):
    """
    Membuat visualisasi missing value sebelum dan sesudah imputasi.
    """
    plot_df = missing_audit.copy()
    plot_df = plot_df[(plot_df["Missing_Sebelum"] > 0) | (plot_df["Missing_Sesudah"] > 0)]

    if plot_df.empty:
        plot_df = missing_audit.copy()

    x = range(len(plot_df))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        [position - width / 2 for position in x],
        plot_df["Missing_Sebelum"],
        width=width,
        label="Sebelum Imputasi",
    )

    plt.bar(
        [position + width / 2 for position in x],
        plot_df["Missing_Sesudah"],
        width=width,
        label="Sesudah Imputasi",
    )

    plt.title("Audit Missing Value")
    plt.xlabel("Kolom")
    plt.ylabel("Jumlah Missing Value")
    plt.xticks(list(x), plot_df["Kolom"], rotation=25, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    output_path = VISUAL_AUDIT_DIR / "audit_missing_value.png"
    save_current_figure(output_path)


def visualize_category_standardization(category_audit: pd.DataFrame):
    """
    Membuat visualisasi kategori sebelum dan sesudah standardisasi untuk setiap kolom kategorikal.
    """
    for column in CATEGORICAL_FEATURES + [TARGET_COLUMN]:
        column_data = category_audit[category_audit["Kolom"] == column].copy()

        if column_data.empty:
            continue

        pivot_data = (
            column_data
            .pivot_table(
                index="Kategori",
                columns="Tahap",
                values="Jumlah",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )

        stages = [col for col in pivot_data.columns if col != "Kategori"]

        x = range(len(pivot_data))
        width = 0.35

        plt.figure(figsize=(10, 5))

        if len(stages) == 1:
            plt.bar(x, pivot_data[stages[0]], width=width, label=stages[0])
        else:
            plt.bar(
                [position - width / 2 for position in x],
                pivot_data[stages[0]],
                width=width,
                label=stages[0],
            )
            plt.bar(
                [position + width / 2 for position in x],
                pivot_data[stages[1]],
                width=width,
                label=stages[1],
            )

        plt.title(f"Audit Standardisasi Kategori - {column}")
        plt.xlabel("Kategori")
        plt.ylabel("Jumlah Data")
        plt.xticks(list(x), pivot_data["Kategori"], rotation=25, ha="right")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        if column == "Penghasilan_Orang_Tua":
            filename = "audit_standardisasi_penghasilan.png"
        elif column == "Kualitas_Internet":
            filename = "audit_standardisasi_internet.png"
        else:
            filename = "audit_standardisasi_status_kelulusan.png"

        output_path = VISUAL_AUDIT_DIR / filename
        save_current_figure(output_path)


def visualize_iqr_audit(iqr_result: pd.DataFrame):
    """
    Membuat visualisasi jumlah outlier sebelum dan sesudah IQR capping.
    """
    x = range(len(iqr_result))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        [position - width / 2 for position in x],
        iqr_result["Outlier_Sebelum"],
        width=width,
        label="Sebelum Capping",
    )

    plt.bar(
        [position + width / 2 for position in x],
        iqr_result["Outlier_Sesudah"],
        width=width,
        label="Sesudah Capping",
    )

    plt.title("Audit Outlier IQR")
    plt.xlabel("Fitur Numerik")
    plt.ylabel("Jumlah Outlier")
    plt.xticks(list(x), iqr_result["Fitur"], rotation=25, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    output_path = VISUAL_AUDIT_DIR / "audit_outlier_iqr.png"
    save_current_figure(output_path)


def visualize_split_data(train_count: int, test_count: int):
    """
    Membuat visualisasi jumlah data training dan testing.
    """
    plt.figure(figsize=(7, 5))

    plt.bar(["Training", "Testing"], [train_count, test_count])

    plt.title("Audit Split Data")
    plt.xlabel("Jenis Data")
    plt.ylabel("Jumlah Data")
    plt.grid(axis="y", alpha=0.3)

    output_path = VISUAL_AUDIT_DIR / "audit_split_data.png"
    save_current_figure(output_path)


# ============================================================
# VISUALISASI EVALUASI MODEL
# ============================================================

def visualize_model_comparison(evaluation_df: pd.DataFrame):
    """
    Membuat visualisasi perbandingan metrik evaluasi Decision Tree dan Random Forest.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]

    x = range(len(metrics))
    width = 0.35

    decision_tree_values = (
        evaluation_df[evaluation_df["Model"] == "Decision Tree"][metrics]
        .iloc[0]
        .values
    )

    random_forest_values = (
        evaluation_df[evaluation_df["Model"] == "Random Forest"][metrics]
        .iloc[0]
        .values
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        [position - width / 2 for position in x],
        decision_tree_values,
        width=width,
        label="Decision Tree",
    )

    plt.bar(
        [position + width / 2 for position in x],
        random_forest_values,
        width=width,
        label="Random Forest",
    )

    plt.title("Perbandingan Evaluasi Model")
    plt.xlabel("Metrik Evaluasi")
    plt.ylabel("Nilai")
    plt.xticks(list(x), metrics)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    output_path = VISUAL_EVALUATION_DIR / "perbandingan_evaluasi_model.png"
    save_current_figure(output_path)


def visualize_confusion_matrix(cm, labels, model_name: str, filename: str):
    """
    Membuat visualisasi confusion matrix sederhana menggunakan matplotlib.
    """
    plt.figure(figsize=(6, 5))

    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()

    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=25, ha="right")
    plt.yticks(tick_marks, labels)

    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text_color = "white" if value > threshold else "black"
            plt.text(
                j,
                i,
                str(value),
                horizontalalignment="center",
                color=text_color,
            )

    plt.ylabel("Aktual")
    plt.xlabel("Prediksi")

    output_path = VISUAL_EVALUATION_DIR / filename
    save_current_figure(output_path)


def visualize_feature_importance(feature_importance_df: pd.DataFrame):
    """
    Membuat visualisasi feature importance dari Random Forest.
    """
    sorted_df = feature_importance_df.sort_values(
        by="Importance",
        ascending=True,
    )

    plt.figure(figsize=(9, 5))

    plt.barh(sorted_df["Feature"], sorted_df["Importance"])

    plt.title("Feature Importance Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Fitur")
    plt.grid(axis="x", alpha=0.3)

    output_path = VISUAL_FEATURE_IMPORTANCE_DIR / "feature_importance_random_forest.png"
    save_current_figure(output_path)