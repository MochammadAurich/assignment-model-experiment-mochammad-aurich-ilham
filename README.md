# AI Model Experiment & Evaluation

### Analisis Sentimen pada Ulasan Pelanggan *E-Commerce*

Repositori ini berisi eksperimen dan evaluasi terhadap dua pendekatan AI yang digunakan untuk menyelesaikan *task* klasifikasi sentimen pada ulasan pelanggan di halaman produk *e-commerce*. Kedua pendekatan yang dibandingkan adalah:

1. **Model Klasik *Machine Learning*** menggunakan Scikit-learn dengan kombinasi TF-IDF dan *Logistic Regression*.
2. **LLM API** menggunakan Gemini dengan pendekatan *few-shot prompting*.



## 1. *Problem Statement* dan *Dataset*

### *Objective*

Sebuah tim produk *e-commerce* ingin menyediakan fitur otomatis untuk mengklasifikasikan sentimen ulasan pelanggan pada halaman produk. Dengan adanya fitur tersebut, proses *monitoring* kepuasan pelanggan tidak lagi harus dilakukan dengan membaca ulasan satu per satu secara manual.

Sebelum fitur tersebut diterapkan pada sistem produksi, dilakukan eksperimen awal untuk membandingkan dua pendekatan AI yang telah ditentukan. Eksperimen ini bertujuan untuk memperoleh rekomendasi mengenai pendekatan yang paling sesuai untuk dikembangkan lebih lanjut.

### *Target/Label*

Target yang diprediksi adalah label sentimen dari setiap ulasan pelanggan. Klasifikasi yang digunakan bersifat biner dengan dua kelas, yaitu `positif` dan `negatif`.

### *Dataset*

*Dataset* yang digunakan adalah `customer_reviews_sentiment.csv`, yang berisi ulasan pelanggan *e-commerce* dengan dua kolom utama:

- `review_text` — teks ulasan pelanggan yang digunakan sebagai fitur atau *input*.
- `sentiment` — label sentimen yang terdiri atas `positif` dan `negatif` sebagai target.

*Dataset* dibagi menjadi data *training* dan data *test* dengan perbandingan 80:20 menggunakan metode *stratified*. *Test set* yang sama digunakan untuk mengevaluasi kedua pendekatan agar hasil perbandingan dapat dilakukan secara adil.

### *Batasan dan Asumsi*

Beberapa batasan dan asumsi yang digunakan dalam eksperimen ini adalah sebagai berikut:

- Tidak dilakukan pencarian maupun penggunaan *dataset* lain di luar *dataset* yang telah diberikan.
- *Dataset* diasumsikan sudah dalam kondisi bersih dan setiap baris hanya memiliki satu label sentimen, tanpa adanya kelas netral.
- Model klasik hanya menggunakan teknik *Machine Learning* konvensional dan tidak menggunakan *Deep Learning*, sesuai dengan cakupan pembelajaran pada sesi ini.
- Pendekatan LLM API menggunakan *prompting* dengan metode *few-shot*, tanpa melakukan proses *fine-tuning* maupun *training* tambahan.



## 2. Ringkasan Eksperimen

### Pendekatan 1 — Model Klasik (Scikit-learn)

Pada pendekatan pertama, digunakan model *Machine Learning* klasik dengan tahapan sebagai berikut:

- Proses *preprocessing* teks dilakukan menggunakan **TF-IDF Vectorizer**.
- Model klasifikasi dilatih menggunakan **Logistic Regression**.
- Prediksi dilakukan pada *test set* yang sama dengan yang digunakan untuk mengevaluasi pendekatan LLM.

### Pendekatan 2 — LLM API (Gemini)

Pendekatan kedua menggunakan LLM melalui Gemini API dengan konfigurasi sebagai berikut:

- Model yang digunakan adalah `gemini-3.5-flash-lite`.
- *Prompt* dirancang menggunakan metode **few-shot prompting** yang terdiri atas empat bagian:
  1. **Persona** — mengarahkan model untuk berperan sebagai asisten analis sentimen yang ahli.
  2. **Task** — memberikan instruksi untuk melakukan klasifikasi sentimen ke dalam dua kelas.
  3. **Konteks** — memberikan empat contoh *few-shot* berupa pasangan antara ulasan dan label.
  4. **Output** — memberikan instruksi agar model hanya menghasilkan satu kata, yaitu `positif` atau `negatif`.
- Parameter `temperature=0.2` digunakan agar hasil yang diberikan model lebih konsisten dan deterministik. Pengaturan ini disesuaikan dengan kebutuhan klasifikasi, bukan untuk *task* generatif yang bersifat kreatif.
- API key yang digunakan berada pada *free tier* dengan *rate limit* sebesar 15 *request* per menit. Oleh karena itu, pemanggilan API terhadap seluruh *test set* dilakukan dengan memberikan jeda antar-*request* serta menggunakan mekanisme *retry* otomatis untuk menangani *error* sementara seperti `429 RESOURCE_EXHAUSTED` dan `503 UNAVAILABLE`.
- *Output* mentah dari LLM kemudian dinormalisasi, seperti mengubah huruf menjadi *lowercase* serta menghapus *whitespace* dan tanda baca, sebelum dibandingkan dengan label asli.



## 3. Tabel Hasil Evaluasi dan Perbandingan

Evaluasi dilakukan menggunakan *test set* yang sama, yaitu sebanyak 40 data yang terdiri atas 18 ulasan negatif dan 22 ulasan positif.

| Metrik | Model Klasik (Scikit-learn) | LLM API (Gemini) |
|---|:---:|:---:|
| *Accuracy* | 1.00 | 1.00 |
| *Precision* | 1.00 | 1.00 |
| *Recall* | 1.00 | 1.00 |
| F1-Score | 1.00 | 1.00 |

Kedua pendekatan berhasil mengklasifikasikan seluruh 40 data *test* dengan benar tanpa adanya kesalahan prediksi. Hal tersebut juga terlihat pada *confusion matrix* kedua model yang menunjukkan hasil sempurna, yaitu `[[18, 0], [0, 22]]`. Dengan demikian, berdasarkan metrik evaluasi yang digunakan, kedua pendekatan tidak menunjukkan perbedaan performa pada eksperimen ini.



## 4. Analisis *Trade-off* dan *Limitation*

### Perbandingan Performa

Kedua pendekatan memperoleh *accuracy* sebesar 1.00. Hasil tersebut kemungkinan dipengaruhi oleh karakteristik *dataset* yang digunakan. Ulasan pelanggan yang terdapat dalam *dataset* cenderung memiliki pola sentimen yang cukup jelas, misalnya melalui penggunaan kata-kata seperti "puas", "kecewa", "lambat", dan "sesuai deskripsi".

Selain itu, ukuran *test set* yang digunakan juga relatif kecil, yaitu hanya 40 data. Oleh karena itu, hasil yang diperoleh belum tentu dapat dipertahankan apabila model diterapkan pada data yang lebih besar dan memiliki karakteristik yang lebih kompleks. Sebagai contoh, model dapat mengalami kesulitan ketika menghadapi ulasan yang memiliki makna ambigu, menggabungkan pujian dan keluhan dalam satu ulasan, atau mengandung sarkasme.

### *Trade-off* *Effort* Implementasi, Kecepatan, dan Biaya

Karena kedua pendekatan menghasilkan performa yang sama pada eksperimen ini, perbandingan selanjutnya dilakukan berdasarkan beberapa faktor lain, yaitu *effort* implementasi, kecepatan inferensi, biaya, dan ketergantungan terhadap layanan eksternal.

| Aspek | Model Klasik (Scikit-learn) | LLM API (Gemini) |
|---|---|---|
| *Effort* implementasi | Membutuhkan proses *training*, termasuk *feature extraction* dan *fitting model*, tetapi implementasinya relatif sederhana | Tidak membutuhkan proses *training*. Model dapat digunakan dengan merancang *prompt* yang sesuai |
| Kecepatan inferensi | Sangat cepat karena berjalan secara lokal dan proses prediksi dapat dilakukan hampir secara instan | Lebih lambat karena bergantung pada *request* melalui jaringan. Pemrosesan 40 data membutuhkan beberapa menit karena adanya jeda antar-*request* |
| Biaya | Tidak terdapat biaya inferensi setelah model selesai dilatih | Pada penggunaan produksi terdapat biaya untuk setiap *request* dan terdapat batasan *rate limit* pada *free tier* |
| Ketergantungan layanan eksternal | Tidak bergantung pada layanan eksternal karena dapat berjalan sepenuhnya secara lokal | Bergantung pada ketersediaan dan performa API dari pihak ketiga |

### *Limitation* yang Ditemukan

Pada Model Klasik, performa sangat dipengaruhi oleh representasi fitur yang digunakan, yaitu TF-IDF, serta jumlah dan keragaman data *training*. Apabila data yang digunakan memiliki variasi yang lebih tinggi, model dapat mengalami kesulitan dalam mengenali kosakata baru yang tidak terdapat pada data *training*. Kondisi tersebut dapat menyebabkan munculnya kata-kata yang bersifat *out-of-vocabulary*.

Sementara itu, pada pendekatan LLM API ditemukan beberapa kendala teknis selama eksperimen, yaitu:

- Terjadi *error* `429 RESOURCE_EXHAUSTED` akibat *rate limit* pada *free tier*, dengan batas maksimal 15 *request* per menit.
- Terjadi *error* `503 UNAVAILABLE` atau *deadline expired* yang berkaitan dengan gangguan sementara pada *server*. Untuk mengatasinya, diperlukan mekanisme *retry* dan jeda antar-*request*.
- *Output* LLM berupa teks bebas sehingga membutuhkan proses normalisasi tambahan sebelum dapat dibandingkan dengan label asli.

Meskipun terdapat beberapa *limitation* pada kedua pendekatan, tidak ditemukan kesalahan prediksi pada eksperimen ini. Oleh karena itu, potensi kesalahan prediksi dari masing-masing pendekatan belum dapat dianalisis lebih lanjut berdasarkan data yang tersedia.



## 5. Rekomendasi *Technical Approach*

Karena kedua pendekatan menghasilkan performa yang sama, yaitu 1.00 pada seluruh metrik evaluasi, rekomendasi didasarkan pada *trade-off* yang ditemukan dari sisi *effort*, kecepatan, biaya, dan *limitation*, bukan berdasarkan perbedaan performa.

**Model Klasik (Scikit-learn) direkomendasikan sebagai pendekatan utama untuk dikembangkan lebih lanjut menuju sistem produksi.**

Pertimbangan tersebut didasarkan pada beberapa hal berikut:

1. **Kecepatan dan biaya operasional**  
   Fitur ini berpotensi digunakan untuk memproses ulasan dalam jumlah besar secara *real-time*. Model klasik yang berjalan secara lokal memiliki proses inferensi yang cepat dan tidak memerlukan biaya untuk setiap *request*. Sementara itu, penggunaan LLM API dalam sistem produksi dapat menimbulkan biaya per *request* serta memiliki batasan *rate limit*.

2. **Tidak bergantung pada layanan eksternal**  
   Selama eksperimen, pendekatan LLM API mengalami beberapa kendala yang berasal dari faktor eksternal, seperti *rate limit* dan *error* sementara pada *server*. Ketergantungan tersebut perlu diperhatikan apabila model digunakan pada sistem produksi yang membutuhkan tingkat *reliability* yang tinggi. Sebaliknya, Model Klasik dapat dijalankan secara lokal sehingga tidak bergantung pada ketersediaan API pihak ketiga.

3. **Performa yang sudah memadai**  
   Pada *dataset* yang digunakan dalam eksperimen ini, Model Klasik mampu mencapai performa yang sama dengan LLM API. Dengan kondisi tersebut, penggunaan pendekatan yang memiliki biaya dan waktu inferensi lebih tinggi belum menunjukkan peningkatan performa pada eksperimen ini.

Meskipun demikian, LLM API tetap dapat digunakan pada kondisi tertentu. Salah satunya adalah sebagai *quick win* pada tahap awal pengembangan produk, terutama ketika jumlah data berlabel belum mencukupi untuk melakukan *training* Model Klasik. LLM juga dapat dipertimbangkan untuk menangani ulasan yang lebih kompleks atau memiliki beberapa konteks dan sentimen yang bercampur.

Performa Model Klasik tetap perlu dipantau ketika model mulai digunakan pada data produksi yang lebih beragam. Apabila ditemukan penurunan performa pada kondisi nyata, pendekatan *hybrid* dapat dipertimbangkan. Dalam pendekatan tersebut, Model Klasik tetap digunakan sebagai model utama, sedangkan LLM dapat berperan sebagai *fallback* untuk kasus yang memiliki tingkat keyakinan prediksi rendah atau sulit diklasifikasikan oleh model utama.



## 6. Cara Menjalankan *Notebook/Script*

### Struktur *Project*

Struktur *project* yang digunakan adalah sebagai berikut:

```text
assignment-model-experiment-mochammad-aurich-ilham-wicaksono/
├── data/
│   └── customer_reviews_sentiment.csv
├── notebook/
│   └── experiment_notebook.ipynb
├── documentation/
│   └── model_comparison_summary.png
├── README.md
└── requirements.txt
```

### Langkah *Setup*

#### 1. *Clone repository* dan masuk ke folder *project*

```bash
git clone <url-repository-ini>
cd assignment-model-experiment-mochammad-aurich-ilham
```

#### 2. Buat dan aktifkan *virtual environment*

```bash
python -m venv venv
```

Untuk Windows menggunakan PowerShell:

```bash
venv\Scripts\Activate.ps1
```

Untuk macOS/Linux:

```bash
source venv/bin/activate
```

#### 3. *Install dependencies*

```bash
pip install -r requirements.txt
```

#### 4. Siapkan file `.env`

File `.env` perlu dibuat pada *root project* dan diisi dengan Gemini API key:

```text
GEMINI_API_KEY=isi_api_key_di_sini
```

API key dapat diperoleh melalui Google AI Studio.

#### 5. Buka *notebook* di VS Code

- Buka `notebook/experiment_notebook.ipynb`.
- Pilih *kernel* Python dari `venv` yang telah dibuat.

#### 6. Jalankan seluruh *cell* secara berurutan

Jalankan seluruh *cell* mulai dari Section 2 hingga Section 6.

Perlu diperhatikan bahwa Section 5.3, yaitu proses pemanggilan Gemini API untuk seluruh *test set*, membutuhkan waktu beberapa menit karena terdapat jeda antar-*request* untuk mematuhi *rate limit* pada *free tier*. Oleh karena itu, proses tersebut sebaiknya tidak dihentikan sebelum selesai.

### *Dependencies* (`requirements.txt`)

```text
pandas
scikit-learn
google-genai
python-dotenv
jupyter
ipykernel
```