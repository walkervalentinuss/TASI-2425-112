<h1 align="center">Analisis Peran Emoji pada Klasifikasi Ujaran Kebencian Dalam Bahasa Indonesia Menggunakan Model IndoBERT dengan Emoji Description</h1>
<p align="center">
  <em>Tugas Akhir - Program Studi Sistem Informasi, Institut Teknologi Del</em>
</p>

---

## 🧠 Latar Belakang

Media sosial telah mempermudah komunikasi, namun juga menjadi sarana penyebaran **ujaran kebencian**, dengan jutaan laporan di platform X (sebelumnya Twitter) di Indonesia. Meskipun model **IndoBERT** telah terbukti efektif dalam deteksi ujaran kebencian berbahasa Indonesia, potensi **emoji** untuk memperkaya makna kontekstual masih belum dimanfaatkan optimal. Penelitian menunjukkan bahwa informasi deskripsi emoji dapat memperkuat sentimen negatif, serta deskripsi tekstual emoji dapat meningkatkan akurasi klasifikasi. Oleh karena itu, penelitian ini bertujuan untuk melatih model IndoBERT dengan **_emoji descriptions_** dan menganalisis pengaruh **_hyperparameter_** terhadap performa model, diharapkan dapat memberikan kontribusi signifikan dalam representasi kontekstual dan memahami peran emoji dalam komunikasi digital.

---

## 🎯 Tujuan Penelitian

✅ Menganalisis peran emoji dalam menambah konteks atau memperjelas makna ujaran kebencian dalam Bahasa Indonesia, serta mengidentifikasi contoh emoji yang dapat memperkuat atau mengubah persepsi terhadap ujaran tersebut.  
✅ Menganalisis pengaruh pengaturan hyperparameter, seperti Batch Size 16 dan 32, Learning Rate (Adam) 5e-5, 3e-5, 2e-5,  dan jumlah Epoch 2, 3, dan 4, terhadap performa model IndoBERT yang dilengkapi emoji description dalam klasifikasi ujaran kebencian berbahasa Indonesia.  
✅ Membandingkan performa model IndoBERT yang dilengkapi emoji description dalam mengklasifikasikan ujaran kebencian bahasa Indonesia yang mengandung emoji, dengan model yang hanya menggunakan teks tanpa emoji.

---

## 🧾 Dataset

Data yang digunakan adalah tweet berbahasa Inggris yang diterjemahkan ke dalam Bahasa Indonesia. Dataset berasal dari:

- **OLID** -> https://huggingface.co/datasets/christophsonntag/OLID
- **TweetEval** –> https://github.com/cardiffnlp/tweeteval/tree/main/datasets  
- **Sushil** –> https://www.researchgate.net/publication/378731934_Enhancing_Hate_Speech_Detection_through_Emoji-based_Classification_using_Bi-LSTM_and_GloVe_Embeddings

---

## ⚙️ Teknologi yang Digunakan

| Komponen            | Teknologi                          |
|---------------------|------------------------------------|
| Model NLP           | IndoBERT-lite-base, mBERT           |
| Tokenisasi          | IndoBERT-lite-base-p1, BertTokenizer                       |
| Emoji Description      | `emoji.demojize()`                  |
| Text Processing     | Sastrawi (Stemming, Stopwords), NLTK, SnowballStemmer     |
| Resampling          | SMOTE                               |
| Evaluasi            | Accuracy, Precision, Recall, F1-Score |

---

## 🔧 Hyperparameter Combination

- **Batch size**: `16`, `32`
- **Learning rate**: `2e-5`, `3e-5`, `5e-5`
- **Epoch**: `2`, `3`, `4`

---

## 🧪 Daftar Eksperimen

Eksperimen terbagi dalam 24 skenario:

1. IndoBERT + Emoji (3 Dataset, Bahasa Indonesia dan Bahasa Inggris)
2. IndoBERT tanpa Emoji (3 Dataset, Bahasa Indonesia dan Bahasa Inggris)
3. mBERT + Emoji (3 Dataset, Bahasa Indonesia dan Bahasa Inggris)
4. mBERT tanpa Emoji (3 Dataset, Bahasa Indonesia dan Bahasa Inggris)

---

## 🧪 Desain Umum & Desain Eksperimen

Desain Umum:
<br>
![Desain Umum](image/TASI-2425-112-V3-DesainUmumPenelitian-V3.png)
<br>

Desain Eksperimen:
<br>
![Desain Eksperimen](image/TASI-2425-112-V3-DesainEksperimen.png)

---

## 🔧 Desain Model IndoBERT dan mBERT

Desain IndoBERT:
<br>
![Desain IndoBERT](image/TASI-2425-112-V3-IndoBERTWE&ED.png)
<br>

Desain mBERT:
<br>
![Desain mBERT](image/TASI-2425-112-V3-mBERTWE&ED.png)

---

## 👩‍💻 Tim Peneliti

- **Walker Valentinus Simanjuntak** – 12S21012  
- **Ruth Marelisa Hutagalung** – 12S21046  
- **Lamria Magdalena Tampubolon** – 12S21055  

📍 Program Studi Sarjana Sistem Informasi  
📚 Fakultas Informatika dan Teknik Elektro  
🏫 Institut Teknologi Del

---
