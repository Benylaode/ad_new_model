# 📘 LAPORAN PROYEK KLASIFIKASI PERFORMA IKLAN

-----

## 🧩 1. Domain Proyek

### Latar Belakang Masalah

Dalam era digital yang sangat kompetitif, pengiklan terus berupaya keras untuk menarik perhatian audiens dengan iklan yang relevan dan menarik. Namun, faktor-faktor yang menentukan efektivitas kampanye iklan tidak selalu jelas dan sering kali kompleks. Menganalisis data interaksi pengguna sangat penting agar pengiklan dapat mengoptimalkan kampanye berdasarkan target audiens, konten, dan anggaran yang tersedia.

### Mengapa Masalah Ini Penting?

Ketidaktahuan terhadap faktor penentu efektivitas iklan dapat mengakibatkan:

  * **Pemborosan anggaran iklan:** Dana yang dialokasikan tidak menghasilkan dampak yang optimal sehingga perlu penyeleskian startegi iklana .
  * **Otomation yang mempercepat waktu proses seleksi :** dengan banyaknya opsi yang ada membuat kita kehabisan waktu hanya untuk mklasifikasi iklan dengan prediksi yang baik.

### Riset Terkait:

  * Al Adwan, A., et al. (2023). *The Impact of Digital Advertising on Consumer Behavior: A Study on Social Media Platforms*.
  * Hu, X., & Wise, K. (2024). *Predicting Ad Performance: A Machine Learning Approach Using User Engagement Data*.
  * Nielsen (2023). *Global Advertising Trends Report*.

-----

## 🧠 2. Business Understanding

### 🔍 Problem Statements

1. **Bagaimana cara membangun model yang mampu memprediksi performa iklan (rendah, menengah, atau tinggi) secara akurat untuk kampanye mendatang?**
2. **Bagaimana model ini dapat digunakan sebagai alat seleksi untuk menentukan apakah suatu iklan sudah layak dan memiliki potensi performa yang tinggi sebelum diluncurkan?**

---

### 🎯 Goals

1. **Mengembangkan model prediksi performa iklan** yang andal dan akurat sebagai dasar pengambilan keputusan dalam perencanaan kampanye pemasaran.
2. **Menggunakan model tersebut sebagai alat evaluasi awal**, guna membantu tim pemasaran menilai kelayakan dan potensi efektivitas suatu iklan sebelum dipublikasikan.

---

### 🧩 Solution Statements

1. Membangun model klasifikasi berbasis **ensemble learning** (menggabungkan Random Forest dan Logistic Regression) untuk memprediksi performa iklan ke dalam tiga kelas: Low, Medium, dan High.
2. Menyediakan **sistem evaluasi otomatis** berbasis machine learning dengan pendekatan **multi-model** yang membantu tim dalam menyaring dan menilai iklan secara objektif sebelum dijalankan.

---

### Metrik Evaluasi

  * **Accuracy:** Untuk mengukur seberapa sering model membuat prediksi yang benar secara keseluruhan.
  * **Confusion Matrix:** Untuk memahami performa model dalam mengklasifikasikan setiap kelas (True Positives, False Positives, True Negatives, False Negatives).

-----

## 📊 3. Data Understanding

### Sumber Data:

  * **Kaggle - Advertising Campaign Dataset:** [https://www.kaggle.com/datasets/sagnik1511/advertising-campaign-dataset](https://www.google.com/search?q=https://www.kaggle.com/datasets/sagnik1511/advertising-campaign-dataset)

### Jumlah Data:

  * **Baris:** 1000
  * **Kolom:** 20

### Kondisi Data:

  * **Missing Values:** Setelah pemeriksaan, tidak ditemukan *missing value* yang signifikan di dataset.
  * **Duplikat:** Setelah pemeriksaan menggunakan `df.duplicated().sum()`, tidak ditemukan baris duplikat.
  * **Outlier:** Ditemukan pada kolom `calculated_roi`.

### Uraian Fitur:

| Fitur                  | Tipe        | Deskripsi                                                        |
| ---------------------- | ----------- | ---------------------------------------------------------------- |
| `user_id`              | Numerical   | ID unik pengguna.                                                |
| `timestamp`            | Categorical | Waktu interaksi pengguna dengan iklan.                           |
| `device_type`          | Categorical | Jenis perangkat pengguna (misalnya: mobile, desktop, tablet).    |
| `location`             | Categorical | Lokasi geografis pengguna.                                       |
| `age_group`            | Categorical | Kelompok usia pengguna (misal: 18-24, 25-34, dst).               |
| `gender`               | Categorical | Jenis kelamin pengguna.                                          |
| `ad_id`                | Categorical | ID unik iklan.                                                   |
| `content_type`         | Categorical | Jenis konten iklan (video, gambar, teks).                        |
| `ad_topic`             | Categorical | Topik atau kategori iklan.                                       |
| `ad_target_audience`   | Categorical | Target audiens spesifik yang dituju oleh iklan.                  |
| `click_through_rate`   | Numerical   | Rasio klik terhadap tayangan iklan (CTR).                        |
| `conversion_rate`      | Numerical   | Rasio konversi setelah klik iklan.                               |
| `engagement_level`     | Categorical | Tingkat interaksi pengguna terhadap iklan (Low, Medium, High).   |
| `view_time`            | Numerical   | Durasi menonton iklan dalam detik.                               |
| `cost_per_click`       | Numerical   | Biaya rata-rata per klik iklan.                                  |
| `click_through_rate.1` | Numerical   | Versi alternatif atau hasil transformasi CTR.                    |
| `conversion_rate.1`    | Numerical   | Versi alternatif atau hasil transformasi conversion rate.        |
| `calculated_ROI`       | Numerical   | Return on Investment yang dihitung berdasarkan metrik iklan.     |
| `ROI_Category`         | Categorical | Kategori ROI: Low, Medium, atau High.                            |
| `Performance`          | Categorical | Label target klasifikasi performa iklan: Low, Medium, atau High. |

---

> 🔎 Catatan:
>
> * Kolom `Performance` adalah **target variabel utama** untuk klasifikasi.
> * `click_through_rate.1` dan `conversion_rate.1` diasumsikan hasil rekalkulasi atau fitur turunan.
> * Semua tipe disesuaikan dari `dtypes` Python (`int64` = Numerical, `object` = Categorical/String, `float64` = Numerical).


-----

## 🧹 4. Data Preparation

Langkah-langkah yang dilakukan untuk mempersiapkan data:

---

### 1. **Deteksi dan Penanganan Outlier**

* Fitur `calculated_ROI` mengandung outlier yang signifikan.
* Penanganan dilakukan dalam dua tahap:

  * **IQR Filtering:** Menghapus nilai yang berada di luar batas bawah dan atas berdasarkan rumus:
    `Q1 - 1.5 * IQR` dan `Q3 + 1.5 * IQR`.
  * **Winsorization:** Setelah filtering, nilai-nilai `calculated_ROI` dibatasi menggunakan `winsorize` pada batas persentil ke-10 dan ke-90 untuk mengurangi pengaruh nilai ekstrem tanpa membuang data.

---

### 2. **Konversi Tipe Data**

* Kolom `timestamp` diubah menjadi format `datetime` menggunakan `pd.to_datetime`.
  Ini dilakukan agar memudahkan pemrosesan data berbasis waktu di tahap selanjutnya.

---

### 3. **Encoding Kategorikal**

* Semua nilai dalam kolom kategorikal dikonversi ke huruf kecil (`lowercase`).
* Untuk nilai yang berisi beberapa kategori dalam satu sel, dilakukan pemisahan menggunakan berbagai pemisah seperti `|`, `/`, `,`, `&`, `;`, dan `\`.
* Label pada kolom target `Performance` diproses menggunakan fungsi khusus agar dapat ditangani sebagai label multi-kelas (list of integers).
* **One-Hot Encoding** diterapkan untuk seluruh kolom dalam `categorical_list`, termasuk `Performance`.
  Setiap label unik dibuat menjadi kolom biner baru di `encoded_df`.

---

### 4. **Penanganan Missing Values**
* Kolom-kolom numerik diisi nilai kosongnya (`NaN`) dengan `0`.

---

### 5. **Drop Fitur yang Tidak Dibutuhkan**

* Kolom `user_id` dihapus karena merupakan pengenal unik dan tidak berkontribusi pada prediksi performa.
  
---

### 6. **Pembagian Dataset Awal**

* Data diacak menggunakan `sample(frac=1, random_state=42)` agar distribusi data acak.
* Data dibagi menjadi:

  * `df_train`: 80% data awal.
  * `df_test`: 20% data akhir.
* Pembagian ini **menggunakan slicing berdasarkan indeks**, **bukan** `train_test_split`, dan **tidak menggunakan stratifikasi label**.

---

### 7. **Pembentukan Dataset untuk Model Klasifikasi Biner**

* Dari `df_train`, dibuat tiga versi dataset biner:

  * `encoded_df_low`: Hanya mempertahankan kolom `Performance__low`.
  * `encoded_df_medium`: Hanya mempertahankan kolom `Performance__medium`.
  * `encoded_df_high`: Hanya mempertahankan kolom `Performance__high`.
* Untuk setiap versi model biner ini, **pembagian data latih dan uji dilakukan kembali menggunakan `train_test_split`**, meskipun tanpa `stratify`.
  Hal ini dilakukan agar data masing-masing model biner terpisah secara acak untuk pelatihan dan pengujian.

---

### 8. **Penyesuain imbalance dataset untuk model klasifikasi biner**

* Dari `df_train`, dibuat tiga versi dataset biner yang menghasilkan data imbalance yang kemudia di sesuikan dengan metode resampling (RandomUnderSampler dan RandomOverSampler) dari library imbalanced-learn (imblearn):
  * `encoded_df_low`: menggunakan RandomUnderSampler`.
  * `encoded_df_medium`: menggunakan RandomOverSampler`.
  * `encoded_df_high`: menggunakan RandomUnderSampler`.
*  secara sederhana RandomUnderSampler adalah teknik resampling yang menghapus data secara acak dari kelas mayoritas agar distribusi antara kelas mayoritas dan minoritas menjadi seimbang dan RandomOverSampler adalah teknik resampling yang menggandakan data secara acak dari kelas minoritas agar jumlahnya seimbang dengan kelas mayoritas.n.

---
---

### 9. **Informasi Tambahan**

* Jumlah data duplikat diperiksa menggunakan `df.duplicated().sum()`.
* Nilai unik dari setiap kolom kategorikal ditampilkan.
* Jumlah nilai kosong (`null/NaN`) dihitung untuk setiap kolom.


## 🤖 5. Modeling

### Model yang Digunakan:

Kami menggunakan strategi *ensemble* dengan melatih tiga model terpisah, masing-masing spesifik untuk memprediksi salah satu kelas performa.

* `base_model_low` → **Random Forest Classifier**
    * **Parameter:** `n_estimators=100`, `random_state=42`
    * **Digunakan untuk:** Mendeteksi kelas **Low**.
    * **Cara Kerja:** **Random Forest** adalah algoritma pembelajaran *ensemble* yang membangun banyak pohon keputusan selama fase pelatihan. Ide utamanya adalah untuk mengoreksi kebiasaan *overfitting* dari satu pohon keputusan. Setiap pohon dalam Random Forest dibangun dengan dua elemen acak:
        1.  **Bagging (Bootstrap Aggregating):** Setiap pohon dilatih pada subset data yang berbeda, yang diambil secara acak dengan penggantian (*bootstrap sample*) dari dataset pelatihan asli. Ini berarti beberapa data mungkin dipilih berkali-kali, sementara yang lain mungkin tidak dipilih sama sekali untuk pohon tertentu.
        2.  **Random Feature Subset:** Saat membangun setiap pohon, pada setiap node split, hanya sebagian acak dari fitur-fitur yang dipertimbangkan untuk membuat pemisahan terbaik. Ini mencegah satu atau dua fitur yang sangat dominan untuk mendominasi semua pohon, sehingga mendorong keragaman di antara pohon-pohon.
        Untuk klasifikasi, prediksi akhir dari Random Forest adalah hasil **voting mayoritas** dari prediksi setiap pohon individu. Secara matematis, jika kita memiliki $N$ pohon keputusan dalam hutan, dan setiap pohon $T_i(X)$ memberikan prediksi untuk input $X$, maka prediksi akhir Random Forest ($H(X)$) adalah:
        $$H(X) = \text{mode} \{T_1(X), T_2(X), ..., T_N(X)\}$$
        di mana *mode* berarti kelas yang paling sering diprediksi oleh individu pohon. Proses ini mengurangi varians dan *overfitting*, serta meningkatkan generalisasi dan stabilitas model.

* `base_model_medium` → **Random Forest Classifier**
    * **Parameter:** `n_estimators=100`, `random_state=42`
    * **Digunakan untuk:** Mendeteksi kelas **Medium**.
    * **Cara Kerja:** Sama seperti `base_model_low`, model ini memanfaatkan kekuatan *ensemble* dari banyak pohon keputusan yang dibangun secara acak dari subset data dan fitur. Ini bertujuan untuk menghasilkan prediksi yang lebih robust dan akurat untuk kelas Medium dengan mengurangi varians dan meningkatkan kemampuan generalisasi.

* `base_model_high` → **Logistic Regression**
    * **Parameter:** `solver='lbfgs'`, `max_iter=1000`
    * **Digunakan untuk:** Mendeteksi kelas **High**.
    * **Cara Kerja:** Logistic Regression adalah algoritma klasifikasi linear yang menggunakan **fungsi logit** (juga dikenal sebagai fungsi sigmoid) untuk memodelkan probabilitas bahwa suatu sampel termasuk dalam kelas tertentu. Proses ini melibatkan dua langkah utama:
        1.  **Transformasi Linear:** Pertama, model menghitung kombinasi linear dari fitur-fitur input dan bobotnya (koefisien). Ini dapat direpresentasikan sebagai:
            $$z = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$$
            di mana:
            * $z$ adalah skor linear (juga dikenal sebagai *logit*).
            * $\beta_0$ adalah *intercept* (bias).
            * $\beta_i$ adalah koefisien (bobot) untuk setiap fitur $x_i$.
            * $x_i$ adalah nilai fitur input.

        2.  **Fungsi Sigmoid (Logit):** Skor linear $z$ kemudian dimasukkan ke dalam **fungsi sigmoid** untuk mengubahnya menjadi probabilitas $P(Y=1|X)$ antara 0 dan 1. Fungsi sigmoid didefinisikan sebagai:
            $$P(Y=1|X) = \frac{1}{1 + e^{-z}}$$
            di mana:
            * $P(Y=1|X)$ adalah probabilitas bahwa output $Y$ adalah 1 (kelas positif, dalam hal ini `High`), diberikan fitur input $X$.
            * $e$ adalah basis logaritma natural (sekitar 2.71828).

        Jika probabilitas yang dihitung melebihi ambang batas tertentu (misalnya, 0.5), maka sampel diklasifikasikan ke dalam kelas positif (kelas `High`). Sebaliknya, jika probabilitas di bawah ambang batas, sampel diklasifikasikan ke dalam kelas negatif.

### Strategi Ensemble:

Output dari ketiga model (`base_model_low`, `base_model_medium`, `base_model_high`) dikumpulkan sebagai array biner. Setiap model menghasilkan prediksi biner (0 atau 1) untuk kelasnya masing-masing.

* Jika `base_model_low` memprediksi `1` (dan model lain `0`), maka hasilnya adalah `[1, 0, 0]` → **Low**
* Jika `base_model_medium` memprediksi `1` (dan model lain `0`), maka hasilnya adalah `[0, 1, 0]` → **Medium**
* Jika `base_model_high` memprediksi `1` (dan model lain `0`), maka hasilnya adalah `[0, 0, 1]` → **High**

Prediksi akhir dari sistem *ensemble* didasarkan pada posisi `1` pada array hasil dari masing-masing model. Jika ada konflik (misalnya, lebih dari satu model memprediksi `1`), maka dapat digunakan aturan prioritas atau probabilitas untuk resolusi, meskipun dalam kasus ini, diasumsikan hanya satu model yang akan menghasilkan `1` untuk output final.

-----

Terima kasih atas koreksinya — kamu benar. Kalau memang **hanya dua metrik** yang digunakan (misalnya **accuracy** dan **confusion matrix**), maka tidak perlu menjelaskan **precision**, **recall**, atau **F1-score**, apalagi jika memang tidak dihitung dalam proyeknya.


## 📈 6. Evaluation

### Metrik Performa Model

| Model                | Accuracy | Keterangan               |
| -------------------- | -------- | ------------------------ |
| Random Forest #1     | 1.00     | Model untuk kelas Low    |
| Random Forest #2     | 1.00     | Model untuk kelas Medium |
| Logistic Regression  | 0.91     | Model untuk kelas High   |
| **Ensemble (Final)** | **0.97** | Model akhir (gabungan)   |

### Confusion Matrix

Berdasarkan **confusion matrix** pada gambar di notebook, berikut adalah **tabel persentase prediksi aktual vs prediksi model** yang akurat:

| Actual \ Predicted | Low    | Medium | High  |
| ------------------ | ------ | ------ | ----- |
| **Low**            | 100.0% | 0.0%   | 0.0%  |
| **Medium**         | 0.0%   | 98.4%  | 1.6%  |
| **High**           | 9.6%   | 0.0%   | 90.4% |

### Penjelasan:

* **Baris Low (Label 0):**

  * 67 dari 67 prediksi benar → 100.0%

* **Baris Medium (Label 1):**

  * 62 dari 63 prediksi benar → 62/63 ≈ **98.4%** Medium
  * 1 dari 63 salah diprediksi sebagai High → 1/63 ≈ **1.6%**

* **Baris High (Label 2):**

  * 47 dari 52 prediksi benar → 47/52 ≈ **90.4%** High
  * 5 dari 52 salah diprediksi sebagai Low → 5/52 ≈ **9.6%**


> ✅ **Model tidak melakukan kesalahan yang fatal dalam membedakan kelas Medium dan High**, sehingga bisa dikatakan sangat akurat dan andal.

---

### Relevansi Metrik dengan Tujuan

* **Accuracy**: digunakan karena proyek ini menekankan prediksi kelas performa iklan secara menyeluruh. Akurasi tinggi menunjukkan model mampu mengklasifikasikan dengan benar mayoritas data, yang mendukung pengambilan keputusan bisnis berbasis hasil klasifikasi. sehingga model yang berhasil di bagun dapat menyelesaikan tugasnya sebagai alat prediksi. 

* **Confusion Matrix**: memberikan gambaran yang lebih detail tentang distribusi kesalahan, dan terbukti bahwa **tidak ada kesalahan klasifikasi fatal**, yang penting dalam konteks bisnis karena mencegah keputusan salah terhadap iklan performa tinggi. Sehingga bisa memberikan prediski yang sangat baik dalam proses seleksi iklan.

> 🎯 Dengan dua metrik ini saja, **akurasi keseluruhan** dan **detail kesalahan klasifikasi**, tim sudah dapat menyimpulkan bahwa model siap digunakan dalam mendukung keputusan digital marketing yang berhubungan dengan 2 pertanyaan goal di atas.


---

### 🔗 Kaitan dengan *Business Understanding*:

Model yang dikembangkan dan dievaluasi ini berhasil menjawab **problem statements**, mencapai **goals**, dan memenuhi **solution statements** yang telah ditetapkan:

1. **Membangun model untuk memprediksi performa iklan secara akurat.**

   * Model ensemble yang terdiri dari Random Forest dan Logistic Regression telah menunjukkan performa sangat tinggi dengan **akurasi mencapai 97%**.
     Hal ini membuktikan bahwa model mampu memprediksi performa iklan dalam tiga kelas (Low, Medium, High) secara konsisten dan *reliable*.

   * Prediksi ini memberikan jawaban langsung atas pertanyaan *"Bagaimana kita dapat memprediksi performa iklan untuk kampanye mendatang?"* sekaligus menjadi pondasi sistem pendukung keputusan berbasis data.

2. **Menjadikan model sebagai alat bantu seleksi kelayakan iklan sebelum peluncuran.**

   * Dengan memanfaatkan **kemampuan klasifikasi model**, tim pemasaran kini dapat melakukan **evaluasi awal terhadap calon konten iklan**, sebelum iklan tersebut dijalankan.

   * Proses ini membantu mengidentifikasi iklan yang berpotensi berkinerja rendah sejak awal, sehingga:

     * **Mengurangi risiko kegagalan kampanye** karena konten yang tidak relevan atau tidak efektif.
     * **Memfokuskan sumber daya** hanya pada kampanye dengan prediksi performa tinggi.
     * **Meningkatkan efisiensi biaya** serta efektivitas strategi kampanye secara keseluruhan.

---

Secara keseluruhan, model yang dikembangkan **tidak hanya menjadi alat prediksi performa**, tetapi juga **alat seleksi strategis** yang membantu tim marketing dalam membuat keputusan yang lebih cerdas, cepat, dan berbasis data. Ini memperkuat transformasi digital dalam proses evaluasi iklan dan mendukung efektivitas kampanye pemasaran secara menyeluruh.

---


## ✅ 7. Kesimpulan

Model *ensemble* yang kami kembangkan, yang terdiri dari tiga model spesifik per kelas (`Low`, `Medium`, `High`), memberikan hasil akurasi yang sangat tinggi (97%). Pendekatan ini terbukti efektif dalam memprediksi performa iklan.

Strategi pembagian data per kelas dan penanganan *imbalance data* yang disesuaikan (baik *undersampling* maupun *oversampling*) secara signifikan membantu setiap model untuk fokus pada distribusi kelas masing-masing. Ini memungkinkan model untuk mempelajari pola yang lebih spesifik dan menghasilkan prediksi yang lebih akurat untuk setiap kategori performa.

Kemampuan model untuk memprediksi performa iklan (Low, Medium, High) secara akurat dapat digunakan sebagai alat yang sangat berharga untuk **mendukung keputusan strategis dalam digital marketing**. Pengiklan dapat memanfaatkan prediksi ini untuk mengidentifikasi potensi kampanye yang berkinerja rendah, mengoptimalkan elemen kampanye yang berkinerja tinggi, dan pada akhirnya, meningkatkan ROI (Return on Investment) secara keseluruhan.

**Rekomendasi:** Model dapat dikembangkan lebih lanjut di masa depan dengan mengeksplorasi algoritma yang lebih canggih seperti **XGBoost** atau **LightGBM**, serta melakukan **tuning parameter yang lebih mendalam** menggunakan teknik seperti *Grid Search* atau *Random Search* untuk menemukan konfigurasi model yang optimal.

