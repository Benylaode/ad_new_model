# 📘 LAPORAN PROYEK KLASIFIKASI PERFORMA IKLAN

-----

## 🧩 1. Domain Proyek

### Latar Belakang Masalah

Dalam era digital yang sangat kompetitif, pengiklan terus berupaya keras untuk menarik perhatian audiens dengan iklan yang relevan dan menarik. Namun, faktor-faktor yang menentukan efektivitas kampanye iklan tidak selalu jelas dan sering kali kompleks. Menganalisis data interaksi pengguna sangat penting agar pengiklan dapat mengoptimalkan kampanye berdasarkan target audiens, konten, dan anggaran yang tersedia.

### Mengapa Masalah Ini Penting?

Ketidaktahuan terhadap faktor penentu efektivitas iklan dapat mengakibatkan:

  * **Pemborosan anggaran iklan:** Dana yang dialokasikan tidak menghasilkan dampak yang optimal.
  * **Kampanye yang tidak tepat sasaran:** Iklan tidak menjangkau audiens yang relevan, sehingga mengurangi potensi konversi.
  * **Format dan konten yang tidak sesuai dengan preferensi pengguna:** Pengguna kurang tertarik atau berinteraksi dengan iklan.

### Riset Terkait:

  * Al Adwan, A., et al. (2023). *The Impact of Digital Advertising on Consumer Behavior: A Study on Social Media Platforms*.
  * Hu, X., & Wise, K. (2024). *Predicting Ad Performance: A Machine Learning Approach Using User Engagement Data*.
  * Nielsen (2023). *Global Advertising Trends Report*.

-----

## 🧠 2. Business Understanding

### Problem Statements

1.  Apakah jenis konten iklan tertentu lebih efektif dalam menghasilkan performa tinggi?
2.  Bagaimana demografi pengguna (usia, lokasi, jenis perangkat) memengaruhi performa iklan?
3.  Bagaimana kita dapat memprediksi performa iklan (rendah, menengah, atau tinggi) untuk kampanye mendatang?

### Goals

1.  Mengidentifikasi jenis **konten iklan** yang paling berkontribusi terhadap performa tinggi.
2.  Menganalisis pengaruh **demografi pengguna** terhadap performa iklan.
3.  Mengembangkan **model prediksi** performa iklan untuk mendukung pengambilan keputusan pemasaran yang lebih baik.

### Solution Statements

1.  Membangun model klasifikasi menggunakan pendekatan **ensemble learning** (terdiri dari Random Forest dan Logistic Regression) untuk memprediksi performa iklan.
2.  Menyediakan **insight berbasis data** kepada pengiklan mengenai faktor-faktor kunci yang memengaruhi performa iklan, sehingga mereka dapat merancang strategi kampanye yang lebih efektif.
3.  Menggunakan pendekatan **multi-model** untuk menangani kompleksitas kelas target (Low, Medium, High) secara spesifik.

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

| Fitur                   | Tipe         | Deskripsi                                                |
| :---------------------- | :----------- | :------------------------------------------------------- |
| `user_id`               | Categorical  | ID unik pengguna.                                        |
| `timestamp`             | String       | Waktu interaksi pengguna dengan iklan.                   |
| `device_type`           | Categorical  | Jenis perangkat yang digunakan pengguna (misalnya, mobile, desktop, tablet). |
| `location`              | Categorical  | Lokasi geografis pengguna.                               |
| `age_group`             | Categorical  | Kelompok usia pengguna (misalnya, 18-24, 25-34).         |
| `gender`                | Categorical  | Jenis kelamin pengguna.                                  |
| `ad_id`                 | Categorical  | ID unik iklan.                                           |
| `content_type`          | Categorical  | Jenis konten iklan (misalnya, video, gambar, teks).      |
| `ad_topic`              | Categorical  | Topik atau kategori iklan.                               |
| `ad_target_audience`    | Categorical  | Target audiens spesifik untuk iklan.                     |
| `click_through_rate`    | Numerical    | Rasio klik terhadap tayangan iklan (CTR).                |
| `conversion_rate`       | Numerical    | Rasio konversi setelah klik iklan.                       |
| `engagement_level`      | Categorical  | Tingkat interaksi pengguna dengan iklan (misalnya, high, medium, low). |
| `view_time`             | Numerical    | Durasi menonton iklan dalam detik.                       |
| `cost_per_click`        | Numerical    | Biaya yang dikeluarkan setiap kali iklan diklik.         |
| `calculated_roi`        | Numerical    | Return on Investment yang dihitung dari kampanye.        |
| `ad_budget`             | Numerical    | Total anggaran yang dialokasikan untuk iklan.            |
| `ad_duration`           | Numerical    | Lama iklan ditayangkan.                                  |
| `previous_campaign_perf`| Categorical  | Performa kampanye sebelumnya untuk pengguna.             |
| `performance`           | Target       | Performa iklan (Low, Medium, High).                      |

-----

## 🧹 4. Data Preparation

Langkah-langkah yang dilakukan untuk mempersiapkan data:

1. **Deteksi dan Penanganan Outlier:**

      * Fitur `calculated_roi` memiliki outlier yang signifikan. Outlier ini ditangani dengan **winsorization** pada batas persentil ke-5 dan ke-95. Penanganan ini dilakukan untuk menghindari bias data yang bisa menyebabkan kesalahan prediksi karena nilai-nilai ekstrem.

3.  **Penghapusan Data Duplikat:**

      * Melakukan penghapusan dengan `df.drop_duplicates()`. Ini dilakukan agar data tidak bias dan condong kepada data yang banyak duplikatnya, memastikan setiap baris data merepresentasikan observasi unik.

4.  **Konversi Tipe Data:**

      * Kolom `timestamp` diubah menjadi tipe data `datetime` dari yang sebelumnya `String`. Ini memudahkan ekstraksi informasi berbasis waktu jika diperlukan di masa depan.

5.  **Drop Fitur yang Tidak Dibutuhkan:**

      * Kolom `user_id` dihapus karena merupakan pengidentifikasi unik dan tidak memberikan kontribusi prediktif yang relevan untuk performa iklan secara umum.

6.  **Encoding Kategorikal:**

      * Menggunakan **One-Hot Encoding** pada kolom-kolom kategorikal seperti `gender`, `device_type`, `content_type`, `location`, `age_group`, `ad_topic`, `ad_target_audience`, `engagement_level`, dan `previous_campaign_perf`. Ini mengubah variabel kategorikal menjadi format numerik yang dapat dipahami oleh model *machine learning*.

7.  **Pembagian Dataset untuk Setiap Model:**

      * Data dibagi berdasarkan kelas target (`performance`) menjadi tiga dataset berbeda. Ini dilakukan karena setiap model klasifikasi (untuk Low, Medium, dan High) akan dilatih secara spesifik untuk mengenali kelas targetnya masing-masing.

8.  **Handling Imbalanced Data:**

      * **Model Low:** Melakukan **undersampling** pada kelas `Medium` dan `High`. Ini mengurangi jumlah sampel dari kelas mayoritas agar seimbang dengan kelas `Low`.
      * **Model Medium:** Melakukan **oversampling** pada kelas `Medium` menggunakan **SMOTE (Synthetic Minority Over-sampling Technique)**. Ini menciptakan sampel sintetis untuk kelas minoritas (`Medium`) guna meningkatkan jumlahnya.
      * **Model High:** Melakukan **undersampling** pada kelas `Medium` dan `Low`. Ini bertujuan untuk menyeimbangkan kelas `High` dengan mengurangi sampel dari kelas lainnya.

9.  **Pembagian Data Latih dan Uji:**

      * Data dibagi menjadi set pelatihan (80%) dan set pengujian (20%) menggunakan `train_test_split` dengan strategi `stratify`. Ini memastikan distribusi kelas di set pelatihan dan pengujian tetap proporsional, yang penting untuk evaluasi model yang akurat.

Tentu, berikut adalah potongan bagian "Modeling" dari laporan Anda, termasuk penjelasan matematis untuk Random Forest dan Logistic Regression:

---

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

## 📈 6. Evaluation

### Metrik Performa Model:

| Model              | Accuracy | Keterangan           |
| :----------------- | :------- | :------------------- |
| Random Forest \#1   | 1.00     | Model untuk kelas Low |
| Random Forest \#2   | 0.99     | Model untuk kelas Medium |
| Logistic Reg.      | 0.91     | Model untuk kelas High |
| **Ensemble (Final)** | **0.97** | Prediksi Akhir       |

### Confusion Matrix dan Analisis:

  * Secara keseluruhan, model mampu membedakan ketiga kelas (`Low`, `Medium`, `High`) dengan sangat baik, seperti yang ditunjukkan oleh akurasi *ensemble* 0.97.
  * **Kesalahan klasifikasi paling banyak terjadi antara kelas `Medium` dan `High`**. Hal ini kemungkinan besar disebabkan oleh distribusi fitur yang saling tumpang tindih antara kedua kelas tersebut, di mana batasan untuk membedakan antara performa "menengah" dan "tinggi" bisa jadi lebih tipis dibandingkan dengan "rendah". Ini menunjukkan bahwa ada beberapa kasus di mana iklan yang sebenarnya berkinerja menengah diklasifikasikan sebagai tinggi, atau sebaliknya.

### Kaitan dengan Business Understanding:

Model yang dikembangkan dan dievaluasi ini berhasil menjawab setiap **problem statement**, mencapai **goals**, dan memenuhi **solution statements** yang telah ditetapkan:

1.  **Mengidentifikasi jenis konten iklan yang paling berkontribusi terhadap performa tinggi.**

      * Model ini, khususnya Random Forest, dapat memberikan informasi mengenai **pentingnya fitur (`feature importance`)**. Dengan menganalisis *feature importance* dari model, kita dapat melihat fitur-fitur seperti `content_type` atau `ad_topic` memiliki bobot yang tinggi dalam memprediksi performa iklan. Contohnya, jika `content_type='video'` dan `ad_topic='fashion'` secara konsisten muncul sebagai fitur dengan bobot penting untuk prediksi kelas `High`, maka kita dapat menyimpulkan bahwa **iklan video dengan topik fashion cenderung berkontribusi terhadap performa tinggi**. Informasi ini sangat berharga bagi pengiklan untuk memfokuskan upaya pembuatan konten.

2.  **Menganalisis pengaruh demografi pengguna terhadap performa iklan.**

      * Sama seperti konten, fitur-fitur demografi seperti `age_group`, `location`, dan `device_type` juga berkontribusi pada prediksi model. Model mampu mengidentifikasi pola di mana demografi tertentu cenderung menunjukkan performa iklan yang berbeda. Misalnya, jika model sering mengklasifikasikan iklan sebagai `High` ketika `age_group='25-34'` dan `device_type='mobile'`, ini menunjukkan bahwa **iklan yang menargetkan kelompok usia 25-34 tahun di perangkat mobile memiliki pengaruh positif terhadap performa tinggi**. Insight ini membantu pengiklan menyempurnakan strategi penargetan audiens mereka.

3.  **Mengembangkan model prediksi performa iklan untuk mendukung pengambilan keputusan pemasaran yang lebih baik.**

      * Dengan **akurasi *ensemble* sebesar 97%**, model ini telah berhasil dikembangkan sebagai alat prediksi yang sangat andal. Akurasi tinggi ini secara langsung mendukung pengambilan keputusan pemasaran yang lebih baik:
          * **Problem Statement Terjawab:** Model ini secara langsung menjawab pertanyaan "Bagaimana kita dapat memprediksi performa iklan?" dengan memberikan prediksi yang akurat untuk kelas `Low`, `Medium`, dan `High`.
          * **Goals Tercapai:** Model ini mencapai *goal* ketiga, yaitu "Mengembangkan model prediksi performa iklan untuk mendukung pengambilan keputusan pemasaran yang lebih baik," dengan menyediakan kemampuan prediksi yang *reliable*.
          * **Solution Statement Berdampak:** Pendekatan *ensemble learning* terbukti efektif dalam menangani klasifikasi multi-kelas, yang merupakan bagian dari *solution statement*. Kemampuan prediksi ini berdampak positif karena pengiklan kini dapat:
              * **Mengalokasikan anggaran secara efisien:** Mereka bisa menghindari pemborosan dana pada kampanye yang diprediksi berkinerja rendah.
              * **Mengoptimalkan strategi:** Mereka dapat mengidentifikasi elemen-elemen sukses (berdasarkan konten dan demografi) dari kampanye yang diprediksi berkinerja tinggi dan mereplikasinya.
              * **Intervensi Dini:** Model dapat mengidentifikasi kampanye yang berpotensi rendah, memungkinkan intervensi cepat untuk penyesuaian strategi.

Secara keseluruhan, wawasan yang diperoleh dari model dan kemampuannya untuk memprediksi performa iklan dapat digunakan untuk **mengoptimalkan strategi kampanye digital marketing** dan **meningkatkan efisiensi biaya** bagi pengiklan.

-----

## ✅ 7. Kesimpulan

Model *ensemble* yang kami kembangkan, yang terdiri dari tiga model spesifik per kelas (`Low`, `Medium`, `High`), memberikan hasil akurasi yang sangat tinggi (97%). Pendekatan ini terbukti efektif dalam memprediksi performa iklan.

Strategi pembagian data per kelas dan penanganan *imbalance data* yang disesuaikan (baik *undersampling* maupun *oversampling*) secara signifikan membantu setiap model untuk fokus pada distribusi kelas masing-masing. Ini memungkinkan model untuk mempelajari pola yang lebih spesifik dan menghasilkan prediksi yang lebih akurat untuk setiap kategori performa.

Kemampuan model untuk memprediksi performa iklan (Low, Medium, High) secara akurat dapat digunakan sebagai alat yang sangat berharga untuk **mendukung keputusan strategis dalam digital marketing**. Pengiklan dapat memanfaatkan prediksi ini untuk mengidentifikasi potensi kampanye yang berkinerja rendah, mengoptimalkan elemen kampanye yang berkinerja tinggi, dan pada akhirnya, meningkatkan ROI (Return on Investment) secara keseluruhan.

**Rekomendasi:** Model dapat dikembangkan lebih lanjut di masa depan dengan mengeksplorasi algoritma yang lebih canggih seperti **XGBoost** atau **LightGBM**, serta melakukan **tuning parameter yang lebih mendalam** menggunakan teknik seperti *Grid Search* atau *Random Search* untuk menemukan konfigurasi model yang optimal.

