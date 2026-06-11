import pandas as pd

from config import (
    AUDIT_DATASET_PATH,
    DUPLICATE_AUDIT_PATH,
    CONFLICT_ID_AUDIT_PATH,
    MISSING_VALUE_AUDIT_PATH,
    IQR_RESULT_PATH,
    FINAL_DATASET_PATH,
    EVALUATION_RESULT_PATH,
    FEATURE_IMPORTANCE_PATH,
    OUTPUT_DIR,
)

from preprocessing import (
    load_dataset,
    create_initial_dataset_audit,
    remove_full_duplicates,
    aggregate_conflict_ids,
    standardize_categories,
    impute_missing_values,
    apply_iqr_capping,
    split_features_target,
)

from visualization import (
    visualize_distributions_before_preprocessing,
    visualize_distributions_after_preprocessing,
    visualize_duplicate_audit,
    visualize_row_count_audit,
    visualize_conflict_id_audit,
    visualize_category_standardization,
    visualize_missing_value_audit,
    visualize_iqr_audit,
)

from modeling import train_and_evaluate_models

from utils import (
    create_project_directories,
    save_dataframe,
    print_section,
)


def main():
    """
    File utama project.

    Alur:
    1. Load dataset.
    2. Audit awal dataset.
    3. Visualisasi distribusi sebelum preprocessing.
    4. Hapus duplikat penuh.
    5. Tangani konflik ID dengan agregasi.
    6. Standardisasi kategori.
    7. Imputasi missing value.
    8. Tangani outlier dengan IQR capping.
    9. Visualisasi distribusi sesudah preprocessing.
    10. Training Decision Tree dan Random Forest.
    11. Evaluasi model.
    12. Simpan seluruh output.
    """

    # ========================================================
    # 1. PERSIAPAN FOLDER
    # ========================================================
    print_section("1. Membuat folder output project")
    create_project_directories()
    print("Folder output berhasil disiapkan.")

    # ========================================================
    # 2. LOAD DATASET
    # ========================================================
    print_section("2. Load dataset")
    dataframe_awal = load_dataset()

    print("Dataset berhasil dibaca.")
    print(f"Jumlah baris awal : {len(dataframe_awal)}")
    print(f"Jumlah kolom      : {dataframe_awal.shape[1]}")

    # ========================================================
    # 3. AUDIT AWAL DATASET
    # ========================================================
    print_section("3. Audit awal dataset")
    audit_dataset_awal = create_initial_dataset_audit(dataframe_awal)
    save_dataframe(audit_dataset_awal, AUDIT_DATASET_PATH)

    print(audit_dataset_awal)
    print(f"Audit awal dataset disimpan ke: {AUDIT_DATASET_PATH}")

    # ========================================================
    # 4. VISUALISASI DISTRIBUSI SEBELUM PREPROCESSING
    # ========================================================
    print_section("4. Visualisasi distribusi sebelum preprocessing")
    visualize_distributions_before_preprocessing(dataframe_awal)

    print("Visualisasi distribusi sebelum preprocessing berhasil dibuat.")

    # ========================================================
    # 5. HAPUS DUPLIKAT PENUH
    # ========================================================
    print_section("5. Menghapus duplikat penuh")
    dataframe_tanpa_duplikat, audit_duplikat = remove_full_duplicates(dataframe_awal)

    save_dataframe(audit_duplikat, DUPLICATE_AUDIT_PATH)
    visualize_duplicate_audit(audit_duplikat)

    print(audit_duplikat)
    print(f"Audit duplikat penuh disimpan ke: {DUPLICATE_AUDIT_PATH}")

    # ========================================================
    # 6. TANGANI KONFLIK ID DENGAN AGREGASI
    # ========================================================
    print_section("6. Menangani konflik ID mahasiswa")
    (
        dataframe_agregasi_id,
        audit_konflik_id,
        detail_konflik_id_sebelum,
        detail_konflik_id_sesudah,
    ) = aggregate_conflict_ids(dataframe_tanpa_duplikat)

    save_dataframe(audit_konflik_id, CONFLICT_ID_AUDIT_PATH)
    save_dataframe(detail_konflik_id_sebelum, OUTPUT_DIR / "detail_konflik_id_sebelum_agregasi.csv")
    save_dataframe(detail_konflik_id_sesudah, OUTPUT_DIR / "detail_konflik_id_sesudah_agregasi.csv")

    visualize_row_count_audit(audit_duplikat, audit_konflik_id)
    visualize_conflict_id_audit(audit_konflik_id)

    print(audit_konflik_id)
    print("Detail konflik ID sebelum agregasi:")
    print(detail_konflik_id_sebelum)
    print("Detail konflik ID sesudah agregasi:")
    print(detail_konflik_id_sesudah)

    # ========================================================
    # 7. STANDARDISASI KATEGORI
    # ========================================================
    print_section("7. Standardisasi kategori")
    dataframe_standard, audit_kategori = standardize_categories(dataframe_agregasi_id)

    save_dataframe(audit_kategori, OUTPUT_DIR / "audit_standardisasi_kategori.csv")
    visualize_category_standardization(audit_kategori)

    print("Standardisasi kategori selesai.")
    print(f"Audit standardisasi kategori disimpan ke: {OUTPUT_DIR / 'audit_standardisasi_kategori.csv'}")

    # ========================================================
    # 8. IMPUTASI MISSING VALUE
    # ========================================================
    print_section("8. Imputasi missing value")
    (
        dataframe_imputasi,
        audit_missing_value,
        ringkasan_imputasi,
    ) = impute_missing_values(dataframe_standard)

    save_dataframe(audit_missing_value, MISSING_VALUE_AUDIT_PATH)
    save_dataframe(ringkasan_imputasi, OUTPUT_DIR / "ringkasan_imputasi.csv")

    visualize_missing_value_audit(audit_missing_value)

    print(audit_missing_value)
    print("Ringkasan imputasi:")
    print(ringkasan_imputasi)

    # ========================================================
    # 9. OUTLIER HANDLING DENGAN IQR CAPPING
    # ========================================================
    print_section("9. Penanganan outlier dengan IQR capping")
    dataframe_final, hasil_iqr = apply_iqr_capping(dataframe_imputasi)

    save_dataframe(hasil_iqr, IQR_RESULT_PATH)
    visualize_iqr_audit(hasil_iqr)

    print(hasil_iqr)
    print(f"Hasil IQR disimpan ke: {IQR_RESULT_PATH}")

    # ========================================================
    # 10. VISUALISASI DISTRIBUSI SESUDAH PREPROCESSING
    # ========================================================
    print_section("10. Visualisasi distribusi sesudah preprocessing")
    visualize_distributions_after_preprocessing(dataframe_final)

    print("Visualisasi distribusi sesudah preprocessing berhasil dibuat.")

    # ========================================================
    # 11. SIMPAN DATASET FINAL SEBELUM ENCODING
    # ========================================================
    print_section("11. Menyimpan dataset setelah preprocessing")
    save_dataframe(dataframe_final, FINAL_DATASET_PATH)

    print(f"Dataset setelah preprocessing disimpan ke: {FINAL_DATASET_PATH}")
    print(f"Jumlah baris dataset final : {len(dataframe_final)}")
    print(f"Jumlah kolom dataset final : {dataframe_final.shape[1]}")

    # ========================================================
    # 12. PEMISAHAN FITUR DAN TARGET
    # ========================================================
    print_section("12. Pemisahan fitur dan target")
    X, y = split_features_target(dataframe_final)

    print(f"Jumlah data fitur  : {X.shape}")
    print(f"Jumlah data target : {y.shape}")

    # ========================================================
    # 13. TRAINING DAN EVALUASI MODEL
    # ========================================================
    print_section("13. Training dan evaluasi model")
    hasil_model = train_and_evaluate_models(X, y)

    evaluation_df = hasil_model["evaluation_df"]
    feature_importance_df = hasil_model["feature_importance_df"]

    save_dataframe(evaluation_df, EVALUATION_RESULT_PATH)
    save_dataframe(feature_importance_df, FEATURE_IMPORTANCE_PATH)

    print("Hasil evaluasi model:")
    print(evaluation_df)

    print("\nFeature importance Random Forest:")
    print(feature_importance_df)

    print(f"\nHasil evaluasi model disimpan ke: {EVALUATION_RESULT_PATH}")
    print(f"Feature importance disimpan ke: {FEATURE_IMPORTANCE_PATH}")

    # ========================================================
    # 14. RINGKASAN OUTPUT
    # ========================================================
    print_section("14. Project selesai dijalankan")
    print("Output utama yang dihasilkan:")
    print("- models/model_decision_tree.pkl")
    print("- models/model_random_forest.pkl")
    print("- models/label_encoder.pkl")
    print("- outputs/audit_dataset_awal.csv")
    print("- outputs/audit_duplikat_penuh.csv")
    print("- outputs/audit_konflik_id.csv")
    print("- outputs/audit_missing_value.csv")
    print("- outputs/hasil_iqr_outlier.csv")
    print("- outputs/dataset_setelah_preprocessing.csv")
    print("- outputs/hasil_evaluasi_model.csv")
    print("- outputs/feature_importance_random_forest.csv")
    print("- outputs/visualisasi/01_distribusi_sebelum_preprocessing/")
    print("- outputs/visualisasi/02_distribusi_sesudah_preprocessing/")
    print("- outputs/visualisasi/03_audit_preprocessing/")
    print("- outputs/visualisasi/04_evaluasi_model/")
    print("- outputs/visualisasi/05_feature_importance/")


if __name__ == "__main__":
    main()