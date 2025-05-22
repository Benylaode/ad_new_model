## 📘 LAPORAN AWAL: Business Understanding & Data Understanding

### 🧩 1. Domain Proyek

**Latar Belakang Masalah**

Dalam era digital yang kompetitif, pengiklan berlomba-lomba untuk menarik perhatian audiens dengan menampilkan iklan yang relevan dan menarik. Namun, sering kali tidak jelas faktor-faktor apa saja yang mempengaruhi efektivitas kampanye iklan. Oleh karena itu, penting untuk menganalisis interaksi pengguna terhadap iklan untuk memahami apa yang membuat kampanye berkinerja tinggi atau rendah. Dengan menganalisis data interaksi ini, pengiklan dapat mengoptimalkan kampanye mereka dari segi target audiens, konten, hingga anggaran.

**Mengapa Masalah Ini Perlu Diselesaikan?**

Ketidaktahuan terhadap variabel kunci yang menentukan efektivitas iklan dapat menyebabkan:

* Pemborosan anggaran pada kampanye yang tidak efektif.
* Target audiens yang tidak tepat sasaran.
* Ketidaksesuaian format atau konten iklan dengan preferensi pengguna.

**Riset Terkait:**

* Al Adwan, A., et al. (2023). *Data analytics in digital marketing for tracking the effectiveness of campaigns and inform strategy*. International Journal of Data and Network Science, 7, 563–574. [https://doi.org/10.5267/j.ijdns.2023.3.015](https://doi.org/10.5267/j.ijdns.2023.3.015)
* Hu, X., & Wise, K. (2024). *Exploring the effects of ad choice and personalization in digital advertising effectiveness*. Journal of Interactive Advertising. [https://doi.org/10.1080/15252019.2024.2375970](https://doi.org/10.1080/15252019.2024.2375970)
* Nielsen. (2023). *How to measure digital audiences and campaigns*. [https://www.nielsen.com/insights/2023/need-to-know-how-to-measure-digital-campaigns-and-audiences/](https://www.nielsen.com/insights/2023/need-to-know-how-to-measure-digital-campaigns-and-audiences/)

---

### 🧠 2. Business Understanding

**Problem Statements**

1. Apakah jenis konten iklan tertentu lebih efektif dalam menghasilkan performa tinggi?
2. Bagaimana demografi pengguna (usia, lokasi, jenis perangkat) memengaruhi performa iklan?
3. Apa ciri khas kampanye dengan performa rendah atau tinggi?

**Goals**

* Mengembangkan model klasifikasi untuk memprediksi kinerja iklan (Low, Medium, High).
* Memberikan wawasan kepada pengiklan agar bisa merancang strategi kampanye yang lebih efektif.

**Solution Statements**

1. Membangun model berbasis **ensemble learning** dengan kombinasi **Random Forest** dan **Logistic Regression** untuk meningkatkan akurasi klasifikasi performa iklan.
2. Menggunakan pendekatan berbasis hasil prediksi biner dari tiap model untuk menghasilkan keputusan akhir yang lebih andal.

**Metrik Evaluasi:**

* Accuracy
* Confusion Matrix

---

### 📊 3. Data Understanding

**Informasi Dataset**

* Jumlah entri: 1000 baris
* Jumlah atribut: 16 kolom
* Tautan data: *(simulasi dataset proyek internal, tidak tersedia publik)*

**Penjelasan Fitur:**

| Kolom                | Tipe        | Deskripsi                 |
| -------------------- | ----------- | ------------------------- |
| user\_id             | Categorical | ID unik pengguna          |
| timestamp            | Datetime    | Waktu interaksi           |
| device\_type         | Categorical | Jenis perangkat           |
| location             | Categorical | Lokasi pengguna           |
| age\_group           | Categorical | Kelompok usia             |
| gender               | Categorical | Jenis kelamin             |
| ad\_id               | Categorical | ID unik iklan             |
| content\_type        | Categorical | Jenis konten              |
| ad\_topic            | Categorical | Topik iklan               |
| ad\_target\_audience | Categorical | Target audiens            |
| click\_through\_rate | Numerical   | Persentase klik           |
| conversion\_rate     | Numerical   | Persentase konversi       |
| engagement\_level    | Categorical | Tingkat interaksi         |
| view\_time           | Numerical   | Lama menonton iklan       |
| cost\_per\_click     | Numerical   | Biaya per klik            |
| performance          | Categorical | Target: Low, Medium, High |

**Distribusi Kelas**

| Kelas  | Jumlah | Persentase |
| ------ | ------ | ---------- |
| High   | 340    | 34.0%      |
| Medium | 330    | 33.0%      |
| Low    | 330    | 33.0%      |

**Visualisasi & Analisis Awal:**

* Data target seimbang, sehingga tidak perlu penyesuaian distribusi kelas.
* Visualisasi fitur numerik menunjukkan outlier pada `cost_per_click`.

  data bisa dilihat pada link ini : https://www.kaggle.com/datasets/ziya07/advertising-campaign-dataset

---

### 🧹 4. Data Preparation

**Langkah-Langkah:**

1. **Pemeriksaan Missing Values**:

   * Tidak ditemukan missing value yang signifikan.
  
2. **Deteksi Outlier**:

   * Fitur `calculated_roi` memiliki outlier yang signifikan dan ditangani dengan winsorization.
     
3. **Hapus data duplikat**:

   * Melakukan penghapusan data duplikat dengan df.duplicated().sum().

3. **Encoding Fitur Kategorikal**:

   * One-Hot Encoding digunakan pada `gender`, `device_type`, `content_type`, dll.

4. **Pembagian Data**:

   * Training: 80%, Testing: 20% menggunakan `train_test_split` dengan `stratify`.

catatan : saya sedikit melakukan penyesuaian dengan colom timestem dan mengubahnya menjadi format datetiem yang seblumnya adalah string

---

### 🤖 5. Modeling

**Model yang Digunakan:**

1. **Random Forest Classifier**

   * Parameter: `n_estimators=100`, `max_depth=10`, `class_weight='balanced'`
   * Kelebihan: Kuat terhadap fitur kategorikal dan tidak sensitif terhadap skala
   * Kekurangan: Interpretasi terbatas

2. **Logistic Regression**

   * Parameter: `C=1.0`, `solver='liblinear'`
   * Kelebihan: Interpretasi koefisien jelas
   * Kekurangan: Kurang efektif untuk relasi non-linear

3. **Voting Classifier**

   * Gabungan dari prediksi model biner untuk tiap kelas (Low, Medium, High)
   * Digunakan teknik stacking sederhana berdasarkan hasil `predict` dari masing-masing model

---

### 📈 6. Evaluation

**Metrik Evaluasi:**

* **Accuracy**: Jumlah prediksi benar dibagi total data
* **Confusion Matrix**: Menilai prediksi tiap kelas

**Hasil Model:**

| Model            | Accuracy | Keterangan       |
| ---------------- | -------- | ---------------- |
| Random Forest #1 | 1.00     | Model individual |
| Random Forest #2 | 0.99     | Model individual |
| Logistic Reg.    | 0.91     | Model individual |
| **Ensemble**     | **0.97** | Model terbaik    |

**Penjelasan Metrik:**

* **Accuracy** dihitung sebagai: $\text{Accuracy} = \frac{TP + TN}{Total}$
* **Confusion Matrix** digunakan untuk mengidentifikasi di mana model salah klasifikasi

---

## ✅ 7. Kesimpulan

* Model ensemble menunjukkan performa tinggi (accuracy 97%).
* Data cukup bersih dan distribusi seimbang.
* Model mampu memberikan prediksi performa yang andal untuk mendukung pengambilan keputusan kampanye iklan.

> Jika diperlukan pengembangan lanjutan, bisa ditambahkan GridSearchCV untuk tuning atau mencoba model lain seperti XGBoost/LightGBM.
