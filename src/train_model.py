import os
import joblib
import pandas as pd
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

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# 2. MEMBACA DATASET
# =====================================================

print("Membaca dataset...")

df = pd.read_excel(DATA_PATH)

print("\nInformasi awal dataset:")
print("Jumlah baris dan kolom:", df.shape)
print("Daftar kolom:", list(df.columns))


# =====================================================
# 3. MENGHAPUS DUPLIKAT PENUH
# =====================================================

jumlah_awal = len(df)
df = df.drop_duplicates()
jumlah_setelah_duplikat = len(df)

print("\nJumlah data awal:", jumlah_awal)
print("Jumlah data setelah hapus duplikat penuh:", jumlah_setelah_duplikat)
print("Jumlah duplikat penuh yang dihapus:", jumlah_awal - jumlah_setelah_duplikat)


# =====================================================
# 4. MENANGANI ID KONFLIK
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


# =====================================================
# 5. STANDARDISASI KATEGORI
# =====================================================

kolom_kategori = [
    "Penghasilan_Orang_Tua",
    "Kualitas_Internet",
    "Status_Kelulusan"
]

for col in kolom_kategori:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

# Mengubah teks kosong/None/Nan menjadi missing value
df = df.replace(["Nan", "None", "", " "], pd.NA)

print("\nCek kategori setelah standardisasi:")
for col in kolom_kategori:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False))


# =====================================================
# 6. MEMISAHKAN FITUR DAN TARGET
# =====================================================

target_col = "Status_Kelulusan"

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


# =====================================================
# 7. MENENTUKAN FITUR NUMERIK DAN KATEGORIKAL
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

# Pastikan kolom numerik benar-benar bertipe numerik
for col in numeric_features:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")


# =====================================================
# 8. MEMBUAT PREPROCESSING PIPELINE
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
# 9. SPLIT DATA LATIH DAN DATA UJI
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
# 10. MEMBUAT MODEL BASELINE
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
# 11. FUNGSI EVALUASI MODEL
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

    plt.tight_layout()

    cm_path = os.path.join(
        OUTPUT_DIR,
        f"confusion_matrix_{nama_model.lower().replace(' ', '_')}.png"
    )

    plt.savefig(cm_path)
    plt.close()

    return hasil_ringkas


# =====================================================
# 12. TRAINING DAN EVALUASI BASELINE
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
# 13. HYPERPARAMETER TUNING
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
# 14. EVALUASI MODEL HASIL TUNING
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
# 15. FEATURE IMPORTANCE RANDOM FOREST TERBAIK
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


# =====================================================
# 16. MENYIMPAN MODEL
# =====================================================

joblib.dump(best_dt_model, os.path.join(MODEL_DIR, "model_decision_tree.pkl"))
joblib.dump(best_rf_model, os.path.join(MODEL_DIR, "model_random_forest.pkl"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("\nModel berhasil disimpan di folder models/")
print("Hasil evaluasi berhasil disimpan di folder outputs/")
print("\nProses training selesai.")