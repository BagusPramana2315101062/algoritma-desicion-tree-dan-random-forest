# Implementasi Decision Tree dan Random Forest untuk Klasifikasi Status Kelulusan Mahasiswa

Project ini merupakan implementasi algoritma **Decision Tree** dan **Random Forest** untuk melakukan klasifikasi status kelulusan mahasiswa. Dataset yang digunakan memuat beberapa fitur akademik dan non-akademik, seperti IPK, kehadiran, jumlah organisasi, total SKS, jam belajar, penghasilan orang tua, dan kualitas internet.

Project ini disusun untuk kebutuhan presentasi materi teori dan implementasi algoritma Decision Tree serta Random Forest dalam mata kuliah Data Mining.

---

## 1. Tujuan Project

Tujuan dari project ini adalah:

1. menjelaskan konsep dasar algoritma Decision Tree;
2. menjelaskan konsep dasar algoritma Random Forest;
3. menerapkan kedua algoritma pada dataset mahasiswa;
4. melakukan preprocessing data secara sistematis;
5. menangani duplikat, konflik ID, missing value, dan outlier;
6. memvisualisasikan distribusi seluruh fitur sebelum dan sesudah preprocessing;
7. membandingkan performa Decision Tree dan Random Forest berdasarkan metrik evaluasi klasifikasi.

---

## 2. Dataset

Dataset yang digunakan berada pada folder:

```text
data/dataset_uts_dirty_preprocessing.xlsx
```

Dataset memiliki kolom sebagai berikut:

| Kolom                 | Jenis       | Keterangan                               |
| --------------------- | ----------- | ---------------------------------------- |
| ID_Mahasiswa          | Identitas   | ID unik mahasiswa                        |
| IPK                   | Numerik     | Indeks prestasi kumulatif mahasiswa      |
| Kehadiran             | Numerik     | Persentase kehadiran mahasiswa           |
| Jumlah_Organisasi     | Numerik     | Jumlah organisasi yang diikuti mahasiswa |
| Total_SKS             | Numerik     | Total SKS yang telah ditempuh            |
| Penghasilan_Orang_Tua | Kategorikal | Tingkat penghasilan orang tua            |
| Jam_Belajar           | Numerik     | Jumlah jam belajar mahasiswa             |
| Kualitas_Internet     | Kategorikal | Kualitas akses internet mahasiswa        |
| Status_Kelulusan      | Target      | Status kelulusan mahasiswa               |

Target klasifikasi pada project ini adalah:

```text
Status_Kelulusan
```

Kelas target terdiri atas:

```text
Tepat Waktu
Tidak Tepat Waktu
```

---

## 3. Algoritma yang Digunakan

### 3.1 Decision Tree

Decision Tree adalah algoritma klasifikasi yang membentuk struktur pohon keputusan. Model ini bekerja dengan membagi data berdasarkan fitur tertentu sehingga menghasilkan aturan keputusan yang mudah dipahami.

Kelebihan Decision Tree:

- mudah dipahami dan divisualisasikan;
- dapat digunakan untuk data numerik dan kategorikal;
- proses prediksinya relatif sederhana.

Kekurangan Decision Tree:

- rentan mengalami overfitting;
- hasil model dapat berubah jika data berubah sedikit;
- kurang stabil dibandingkan metode ensemble.

### 3.2 Random Forest

Random Forest adalah algoritma ensemble yang membangun banyak Decision Tree, kemudian menggabungkan hasil prediksinya. Dengan menggunakan banyak pohon keputusan, Random Forest umumnya lebih stabil dibandingkan Decision Tree tunggal.

Kelebihan Random Forest:

- lebih stabil dibandingkan Decision Tree tunggal;
- mampu mengurangi risiko overfitting;
- dapat memberikan nilai feature importance;
- cocok digunakan untuk masalah klasifikasi.

Kekurangan Random Forest:

- lebih sulit diinterpretasikan dibandingkan Decision Tree;
- membutuhkan waktu komputasi lebih besar;
- hasil model tidak sesederhana satu pohon keputusan.

---

## 4. Tahapan Project

Alur utama project dijalankan melalui file:

```text
src/main.py
```

Tahapan yang dilakukan adalah:

1. membaca dataset;
2. melakukan audit awal dataset;
3. membuat visualisasi distribusi seluruh fitur sebelum preprocessing;
4. menghapus duplikat penuh;
5. menangani konflik ID mahasiswa menggunakan agregasi;
6. melakukan standardisasi kategori;
7. melakukan imputasi missing value;
8. mendeteksi dan menangani outlier menggunakan metode IQR capping;
9. membuat visualisasi distribusi seluruh fitur sesudah preprocessing;
10. memisahkan fitur dan target;
11. melakukan encoding target dan fitur kategorikal;
12. membagi data menjadi data training dan testing;
13. melatih model Decision Tree;
14. melatih model Random Forest;
15. mengevaluasi model;
16. menyimpan hasil evaluasi, visualisasi, dan model.

---

## 5. Preprocessing Data

### 5.1 Penghapusan Duplikat Penuh

Duplikat penuh adalah baris data yang seluruh nilainya sama persis pada semua kolom. Pada project ini, duplikat penuh dihapus karena hanya merupakan salinan identik dan tidak menambah variasi informasi pada dataset.

### 5.2 Penanganan Konflik ID Mahasiswa

Konflik ID terjadi ketika satu `ID_Mahasiswa` muncul lebih dari satu kali dengan nilai fitur yang berbeda. Konflik ID tidak dihapus secara langsung, tetapi ditangani dengan agregasi per ID mahasiswa.

Aturan agregasi yang digunakan:

| Jenis Kolom | Metode |
| ----------- | ------ |
| Numerik     | Median |
| Kategorikal | Modus  |
| Target      | Modus  |

Dengan metode ini, setiap mahasiswa direpresentasikan oleh satu record final.

### 5.3 Standardisasi Kategori

Kolom kategorikal distandardisasi agar format penulisannya seragam. Proses ini dilakukan dengan menghapus spasi berlebih dan menyeragamkan format huruf.

Kolom yang distandardisasi:

```text
Penghasilan_Orang_Tua
Kualitas_Internet
Status_Kelulusan
```

### 5.4 Imputasi Missing Value

Missing value ditangani menggunakan metode berikut:

| Jenis Data  | Metode Imputasi |
| ----------- | --------------- |
| Numerik     | Median          |
| Kategorikal | Modus           |

Target kosong tidak diimputasi karena target merupakan label yang akan diprediksi.

### 5.5 Penanganan Outlier dengan IQR Capping

Metode **Interquartile Range (IQR)** digunakan untuk mendeteksi dan menangani outlier pada fitur numerik.

Rumus yang digunakan:

```text
IQR = Q3 - Q1
Batas bawah = Q1 - 1.5 × IQR
Batas atas = Q3 + 1.5 × IQR
```

Outlier tidak dihapus, tetapi ditangani dengan metode **capping**:

- nilai yang lebih kecil dari batas bawah diganti menjadi batas bawah;
- nilai yang lebih besar dari batas atas diganti menjadi batas atas.

Pendekatan ini digunakan agar jumlah data tetap terjaga, tetapi pengaruh nilai ekstrem terhadap model dapat dikurangi.

---

## 6. Visualisasi Data

Visualisasi dibuat dalam dua tahap utama:

### 6.1 Visualisasi Sebelum Preprocessing

Visualisasi ini digunakan untuk melihat kondisi awal data sebelum dilakukan pembersihan dan transformasi.

Folder output:

```text
outputs/visualisasi/01_distribusi_sebelum_preprocessing
```

Visualisasi yang dibuat:

```text
distribusi_ipk.png
distribusi_kehadiran.png
distribusi_jumlah_organisasi.png
distribusi_total_sks.png
distribusi_jam_belajar.png
distribusi_penghasilan_orang_tua.png
distribusi_kualitas_internet.png
distribusi_status_kelulusan.png
```

### 6.2 Visualisasi Sesudah Preprocessing

Visualisasi ini digunakan untuk melihat kondisi data setelah dilakukan duplikat handling, konflik ID handling, standardisasi kategori, imputasi missing value, dan IQR capping.

Folder output:

```text
outputs/visualisasi/02_distribusi_sesudah_preprocessing
```

Visualisasi yang dibuat:

```text
distribusi_ipk.png
distribusi_kehadiran.png
distribusi_jumlah_organisasi.png
distribusi_total_sks.png
distribusi_jam_belajar.png
distribusi_penghasilan_orang_tua.png
distribusi_kualitas_internet.png
distribusi_status_kelulusan.png
```

Pada visualisasi distribusi, label status kelulusan diberi warna berbeda agar lebih mudah dipresentasikan:

| Label                           | Warna |
| ------------------------------- | ----- |
| Tepat Waktu / Lulus             | Biru  |
| Tidak Tepat Waktu / Tidak Lulus | Merah |

### 6.3 Visualisasi Audit Preprocessing

Visualisasi audit digunakan untuk menunjukkan perubahan data pada setiap tahap preprocessing.

Folder output:

```text
outputs/visualisasi/03_audit_preprocessing
```

Visualisasi audit meliputi:

```text
audit_jumlah_baris.png
audit_duplikat_penuh.png
audit_konflik_id.png
audit_missing_value.png
audit_standardisasi_penghasilan.png
audit_standardisasi_internet.png
audit_standardisasi_status_kelulusan.png
audit_outlier_iqr.png
audit_split_data.png
```

### 6.4 Visualisasi Evaluasi Model

Folder output:

```text
outputs/visualisasi/04_evaluasi_model
```

Visualisasi evaluasi meliputi:

```text
perbandingan_evaluasi_model.png
confusion_matrix_decision_tree.png
confusion_matrix_random_forest.png
```

### 6.5 Visualisasi Feature Importance

Folder output:

```text
outputs/visualisasi/05_feature_importance
```

Visualisasi yang dibuat:

```text
feature_importance_random_forest.png
```

Feature importance menunjukkan kontribusi relatif setiap fitur dalam membantu model Random Forest melakukan prediksi. Nilai ini bukan korelasi dan tidak menunjukkan hubungan sebab-akibat.

---

## 7. Struktur Folder Project

Struktur utama project:

```text
algoritma-desicion-tree-dan-random-forest/
│
├── data/
│   └── dataset_uts_dirty_preprocessing.xlsx
│
├── models/
│   ├── label_encoder.pkl
│   ├── model_decision_tree.pkl
│   └── model_random_forest.pkl
│
├── outputs/
│   └── visualisasi/
│       ├── 01_distribusi_sebelum_preprocessing/
│       ├── 02_distribusi_sesudah_preprocessing/
│       ├── 03_audit_preprocessing/
│       ├── 04_evaluasi_model/
│       └── 05_feature_importance/
│
├── src/
│   ├── config.py
│   ├── main.py
│   ├── modeling.py
│   ├── preprocessing.py
│   ├── utils.py
│   └── visualization.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 8. Struktur Kode

### 8.1 `src/main.py`

File utama untuk menjalankan seluruh alur project dari awal sampai akhir.

### 8.2 `src/config.py`

Berisi konfigurasi path, nama kolom, daftar fitur numerik, daftar fitur kategorikal, target, dan lokasi output.

### 8.3 `src/preprocessing.py`

Berisi fungsi untuk:

- load dataset;
- audit awal dataset;
- hapus duplikat penuh;
- agregasi konflik ID;
- standardisasi kategori;
- imputasi missing value;
- IQR capping;
- pemisahan fitur dan target.

### 8.4 `src/visualization.py`

Berisi fungsi untuk membuat:

- distribusi fitur sebelum preprocessing;
- distribusi fitur sesudah preprocessing;
- audit preprocessing;
- evaluasi model;
- confusion matrix;
- feature importance.

### 8.5 `src/modeling.py`

Berisi fungsi untuk:

- encoding target;
- preprocessing fitur;
- training Decision Tree;
- training Random Forest;
- evaluasi model;
- classification report;
- feature importance;
- penyimpanan model.

### 8.6 `src/utils.py`

Berisi fungsi bantu untuk membuat folder output, menyimpan DataFrame, dan menampilkan judul proses di terminal.

---

## 9. Cara Menjalankan Project

### 9.1 Aktifkan Virtual Environment

```cmd
.venv\Scripts\activate
```

### 9.2 Install Dependensi

```cmd
python -m pip install -r requirements.txt
```

### 9.3 Jalankan Program

```cmd
python src\main.py
```

---

## 10. Output Project

Setelah program dijalankan, output utama akan tersimpan pada folder `models` dan `outputs`.

### 10.1 Output Model

```text
models/model_decision_tree.pkl
models/model_random_forest.pkl
models/label_encoder.pkl
```

### 10.2 Output Visualisasi

```text
outputs/visualisasi/
```

Gambar yang digunakan untuk PPT sebaiknya diambil dari folder `outputs/visualisasi`, terutama:

```text
outputs/visualisasi/01_distribusi_sebelum_preprocessing
outputs/visualisasi/02_distribusi_sesudah_preprocessing
outputs/visualisasi/03_audit_preprocessing
outputs/visualisasi/04_evaluasi_model
outputs/visualisasi/05_feature_importance
```

### 10.3 Output Data Audit

Program juga menghasilkan file data audit seperti CSV dan TXT. File ini digunakan sebagai bukti pendukung proses, bukan sebagai bahan utama PPT.

Contoh output data audit:

```text
audit_dataset_awal.csv
audit_duplikat_penuh.csv
audit_konflik_id.csv
audit_missing_value.csv
hasil_iqr_outlier.csv
hasil_evaluasi_model.csv
classification_report_decision_tree.txt
classification_report_random_forest.txt
```

---

## 11. Evaluasi Model

Model dievaluasi menggunakan metrik:

| Metrik           | Keterangan                                                       |
| ---------------- | ---------------------------------------------------------------- |
| Accuracy         | Mengukur proporsi prediksi benar dari seluruh data uji           |
| Precision        | Mengukur ketepatan prediksi pada masing-masing kelas             |
| Recall           | Mengukur kemampuan model menemukan data pada masing-masing kelas |
| F1-score         | Menggabungkan precision dan recall dalam satu ukuran             |
| Confusion Matrix | Menampilkan jumlah prediksi benar dan salah per kelas            |

Perbandingan hasil Decision Tree dan Random Forest divisualisasikan pada:

```text
outputs/visualisasi/04_evaluasi_model/perbandingan_evaluasi_model.png
```

---

## 12. Catatan Feature Importance

Feature importance digunakan untuk melihat fitur mana yang paling banyak berkontribusi dalam proses prediksi model Random Forest.

Feature importance tidak sama dengan korelasi. Korelasi mengukur hubungan statistik antarvariabel, sedangkan feature importance menunjukkan kontribusi fitur terhadap keputusan model.

Nilai feature importance juga tidak dapat diartikan sebagai hubungan sebab-akibat.

---

## 13. Kesimpulan Project

Project ini menunjukkan bahwa algoritma Decision Tree dan Random Forest dapat digunakan untuk klasifikasi status kelulusan mahasiswa. Proses preprocessing menjadi bagian penting karena dataset memiliki duplikat penuh, konflik ID, missing value, variasi kategori, dan outlier numerik.

Decision Tree memberikan model yang lebih mudah dijelaskan, sedangkan Random Forest memberikan pendekatan ensemble yang lebih stabil. Visualisasi sebelum dan sesudah preprocessing membantu menunjukkan perubahan data secara lebih jelas, terutama untuk kebutuhan presentasi.

---
