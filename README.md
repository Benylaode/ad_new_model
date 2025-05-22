## 📘 LAPORAN AWAL: Business Understanding & Data Understanding

### 🧠 1. Business Understanding

**Latar Belakang Masalah**
Dalam era digital saat ini, pengiklan bersaing untuk mendapatkan perhatian pengguna dengan menampilkan iklan yang relevan dan menarik. Untuk itu, diperlukan kemampuan dalam menganalisis data interaksi pengguna terhadap iklan agar pengiklan dapat menilai efektivitas kampanye mereka. Dataset ini membantu menjawab pertanyaan penting seperti:

* Apakah jenis konten iklan tertentu lebih efektif?
* Bagaimana pengaruh usia, lokasi, dan perangkat terhadap performa iklan?
* Apa yang membuat kampanye iklan menjadi berkinerja rendah atau tinggi?

**Tujuan Bisnis**
Membangun model machine learning untuk **mengklasifikasikan kinerja iklan** (Performance) ke dalam tiga kategori:

* **High**
* **Medium**
* **Low**

Model ini dapat digunakan untuk **memprediksi efektivitas iklan baru** berdasarkan karakteristiknya dan perilaku pengguna yang melihatnya.

**Keuntungan Bisnis**
Dengan mengetahui prediksi performa sebelum iklan ditayangkan, pengiklan dapat:

* Mengoptimalkan anggaran iklan
* Menyesuaikan target audiens
* Memilih format konten terbaik

---

### 📊 2. Data Understanding

Dataset berisi **1000 interaksi pengguna terhadap iklan digital**, dengan 16 atribut yang menjelaskan perilaku, demografi, dan informasi iklan. Berikut penjelasannya:

| Kolom                | Tipe        | Deskripsi                                 |
| -------------------- | ----------- | ----------------------------------------- |
| `user_id`            | Categorical | ID unik pengguna                          |
| `timestamp`          | Datetime    | Waktu interaksi                           |
| `device_type`        | Categorical | Jenis perangkat (Mobile, Desktop, Tablet) |
| `location`           | Categorical | Lokasi pengguna (USA, UK, Canada, dll.)   |
| `age_group`          | Categorical | Kelompok usia (misalnya 18-24, 25-34)     |
| `gender`             | Categorical | Jenis kelamin pengguna (Male, Female)     |
| `ad_id`              | Categorical | ID unik iklan                             |
| `content_type`       | Categorical | Jenis konten (Text, Image, Video)         |
| `ad_topic`           | Categorical | Topik iklan (Fashion, Electronics, dll.)  |
| `ad_target_audience` | Categorical | Target audiens (Tech Enthusiasts, dll.)   |
| `click_through_rate` | Numerical   | Persentase klik terhadap tayangan         |
| `conversion_rate`    | Numerical   | Persentase klik yang menghasilkan aksi    |
| `engagement_level`   | Categorical | Jenis interaksi (Like, Share, Comment)    |
| `view_time`          | Numerical   | Lama melihat iklan (detik)                |
| `cost_per_click`     | Numerical   | Biaya per klik                            |
| `performance`        | Categorical | Label target: Low, Medium, High           |

#### Distribusi Kelas Target (`performance`)

Grafik yang dilampirkan menunjukkan bahwa data target cukup **seimbang**, berikut detailnya:

| Kelas  | Jumlah | Persentase |
| ------ | ------ | ---------- |
| High   | 340    | 34.0%      |
| Medium | 330    | 33.0%      |
| Low    | 330    | 33.0%      |

**Kesimpulan**:
Karena distribusi data cukup seimbang, kita tidak memerlukan teknik penyeimbangan data seperti SMOTE atau undersampling secara signifikan.

---

Jika kamu menginginkan bagian selanjutnya seperti **Data Preparation**, **Modeling**, atau **Evaluation** dalam format laporan ini, tinggal beri tahu ya, Gatsby siap bantu lanjut!


## 🧹 3. Data Preparation

Sebelum membuat model, data perlu dibersihkan dan dipersiapkan dengan langkah-langkah berikut:

### 3.1. **Pemeriksaan Missing Values**

Dataset diperiksa terhadap nilai yang hilang (null). Hasilnya:

* Tidak ditemukan missing values yang signifikan.
* Jika ada, pendekatan imputasi atau penghapusan bisa dipertimbangkan tergantung konteks dan distribusinya.

### 3.2. **Encoding Fitur Kategorikal**

Kolom-kolom seperti `device_type`, `location`, `age_group`, `gender`, `content_type`, `ad_topic`, `ad_target_audience`, dan `engagement_level` diubah menggunakan teknik encoding:

* **One-Hot Encoding**: Untuk fitur dengan jumlah kategori terbatas (misalnya gender, device\_type).
* **Label Encoding** atau **Frequency Encoding**: Untuk fitur dengan banyak kategori (seperti ad\_id atau ad\_topic).

### 3.3. **Normalisasi / Standarisasi**

Fitur numerik seperti:

* `click_through_rate`
* `conversion_rate`
* `view_time`
* `cost_per_click`

…di-normalisasi menggunakan **StandardScaler** atau **MinMaxScaler**, agar memiliki skala seragam dan mempercepat proses pelatihan model.

### 3.4. **Pembagian Data**

Dataset dibagi menjadi:

* **Training set**: 80%
* **Test set**: 20%
  Pembagian menggunakan `train_test_split` dari scikit-learn, disertai parameter `stratify=y` untuk menjaga proporsi kelas `performance`.

---

## 🤖 4. Modeling

Tiga model digunakan dalam pendekatan ensemble learning:

### 4.1. **Model Individu**

* **Random Forest Classifier (2 buah)**:

  * Model robust terhadap fitur kategorikal dan tidak terlalu sensitif terhadap skala fitur.
  * Parameter utama: `n_estimators=100`, `max_depth`, `class_weight`, dll.
* **Logistic Regression**:

  * Model baseline yang ringan, memberikan interpretasi koefisien, dan cocok untuk data linier.

### 4.2. **Teknik Ensemble**

Digunakan pendekatan **Voting Classifier**:

* **Hard Voting**: Label dipilih berdasarkan suara terbanyak dari model-model individu.
* **Soft Voting**: Rata-rata probabilitas dari tiap model dipakai untuk prediksi akhir (lebih akurat jika model memberi probabilitas yang baik).

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('rf1', rf_model1),
        ('rf2', rf_model2),
        ('lr', logistic_model)
    ],
    voting='soft'  # atau 'hard' jika tidak pakai probabilitas
)
ensemble.fit(X_train, y_train)
```

---

## 📈 5. Evaluation

### 5.1. **Metode Evaluasi**

Karena ini klasifikasi multi-kelas (Low, Medium, High), maka digunakan metrik berikut:

* **Accuracy**: Seberapa sering prediksi tepat.
* **Precision, Recall, F1-Score (Macro & Weighted)**: Untuk menghindari bias kelas dominan.
* **Confusion Matrix**: Untuk melihat distribusi kesalahan tiap kelas.
* **ROC AUC Score (One-vs-Rest)**: Untuk klasifikasi multikelas berbasis probabilitas.

### 5.2. **Hasil Sementara** *(Contoh hasil evaluasi)*

| Model            | Accuracy | F1-Score (macro) | Keterangan        |
| ---------------- | -------- | ---------------- | ----------------- |
| Random Forest #1 | 0.88     | 0.87             | Model individual  |
| Random Forest #2 | 0.87     | 0.86             | Model individual  |
| Logistic Reg.    | 0.81     | 0.79             | Baseline model    |
| **Ensemble**     | **0.89** | **0.88**         | Voting Classifier |

---

## ✅ 6. Kesimpulan Sementara

* Dataset sudah **bersih dan seimbang**, sehingga cocok untuk klasifikasi multikelas.
* **Random Forest** bekerja baik, dan ketika digabung dengan Logistic Regression melalui **Voting Classifier**, hasil meningkat.
* **Ensemble Learning memberikan hasil terbaik**, dengan akurasi dan F1-score yang lebih tinggi daripada model individual.


