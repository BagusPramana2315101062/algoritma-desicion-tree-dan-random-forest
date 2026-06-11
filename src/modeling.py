import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    DECISION_TREE_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    LABEL_ENCODER_PATH,
    OUTPUT_DIR,
)

from visualization import (
    visualize_model_comparison,
    visualize_confusion_matrix,
    visualize_feature_importance,
    visualize_split_data,
)


# ============================================================
# ENCODING TARGET
# ============================================================

def encode_target(y: pd.Series):
    """
    Melakukan encoding pada target Status_Kelulusan.

    Contoh:
    - Tidak Tepat Waktu -> 0
    - Tepat Waktu -> 1

    Urutan angka mengikuti hasil LabelEncoder.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return y_encoded, label_encoder


# ============================================================
# PREPROCESSOR FITUR
# ============================================================

def build_feature_preprocessor():
    """
    Membuat preprocessing fitur sebelum masuk ke model.

    Catatan:
    - Missing value sudah ditangani pada preprocessing.py.
    - Fitur numerik langsung diteruskan.
    - Fitur kategorikal diubah menjadi angka menggunakan OrdinalEncoder.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# MODEL
# ============================================================

def build_decision_tree_model():
    """
    Membuat pipeline model Decision Tree.
    """
    preprocessor = build_feature_preprocessor()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


def build_random_forest_model():
    """
    Membuat pipeline model Random Forest.
    """
    preprocessor = build_feature_preprocessor()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=7,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


# ============================================================
# EVALUASI
# ============================================================

def evaluate_single_model(model_name: str, model, X_test, y_test) -> dict:
    """
    Mengevaluasi satu model klasifikasi.
    """
    y_pred = model.predict(X_test)

    evaluation = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    return evaluation


def create_classification_report_text(model_name: str, model, X_test, y_test, label_encoder):
    """
    Membuat classification report dalam bentuk teks.
    """
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    output_path = OUTPUT_DIR / f"classification_report_{model_name.lower().replace(' ', '_')}.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"Classification Report - {model_name}\n")
        file.write("=" * 60)
        file.write("\n")
        file.write(report)


def create_confusion_matrix_output(model_name: str, model, X_test, y_test, label_encoder, filename: str):
    """
    Membuat confusion matrix dan menyimpannya sebagai gambar.
    """
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    visualize_confusion_matrix(
        cm=cm,
        labels=label_encoder.classes_,
        model_name=model_name,
        filename=filename,
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_names_after_preprocessing():
    """
    Mengambil nama fitur setelah preprocessing.

    Karena categorical encoding menggunakan OrdinalEncoder,
    jumlah fitur tidak bertambah. Nama fitur tetap:
    numerik + kategorikal.
    """
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_random_forest_feature_importance(random_forest_model) -> pd.DataFrame:
    """
    Membuat tabel feature importance dari model Random Forest.
    """
    classifier = random_forest_model.named_steps["classifier"]
    importances = classifier.feature_importances_

    feature_names = get_feature_names_after_preprocessing()

    feature_importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    )

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance",
        ascending=False,
    )

    return feature_importance_df


# ============================================================
# TRAINING DAN EVALUASI UTAMA
# ============================================================

def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series):
    """
    Melakukan seluruh proses modeling:
    - encoding target,
    - split train-test,
    - training Decision Tree,
    - training Random Forest,
    - evaluasi model,
    - visualisasi evaluasi,
    - feature importance,
    - penyimpanan model.
    """
    y_encoded, label_encoder = encode_target(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    visualize_split_data(
        train_count=len(X_train),
        test_count=len(X_test),
    )

    decision_tree_model = build_decision_tree_model()
    random_forest_model = build_random_forest_model()

    decision_tree_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    evaluation_rows = [
        evaluate_single_model(
            model_name="Decision Tree",
            model=decision_tree_model,
            X_test=X_test,
            y_test=y_test,
        ),
        evaluate_single_model(
            model_name="Random Forest",
            model=random_forest_model,
            X_test=X_test,
            y_test=y_test,
        ),
    ]

    evaluation_df = pd.DataFrame(evaluation_rows)

    visualize_model_comparison(evaluation_df)

    create_confusion_matrix_output(
        model_name="Decision Tree",
        model=decision_tree_model,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder,
        filename="confusion_matrix_decision_tree.png",
    )

    create_confusion_matrix_output(
        model_name="Random Forest",
        model=random_forest_model,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder,
        filename="confusion_matrix_random_forest.png",
    )

    create_classification_report_text(
        model_name="Decision Tree",
        model=decision_tree_model,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder,
    )

    create_classification_report_text(
        model_name="Random Forest",
        model=random_forest_model,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder,
    )

    feature_importance_df = create_random_forest_feature_importance(random_forest_model)

    visualize_feature_importance(feature_importance_df)

    joblib.dump(decision_tree_model, DECISION_TREE_MODEL_PATH)
    joblib.dump(random_forest_model, RANDOM_FOREST_MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    return {
        "decision_tree_model": decision_tree_model,
        "random_forest_model": random_forest_model,
        "label_encoder": label_encoder,
        "evaluation_df": evaluation_df,
        "feature_importance_df": feature_importance_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }