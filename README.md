<h1 align="center">📚 Analisis Peran Emoji dalam Klasifikasi Ujaran Kebencian Dalam Bahasa Indonesia Menggunakan Model IndoBERT dengan Emoji Description</h1>
<p align="center">
  <em>Tugas Akhir - Program Studi Sistem Informasi, Institut Teknologi Del</em>
</p>


---

## 🧠 Latar Belakang

Emoji adalah bagian penting dari komunikasi digital modern. Namun, emoji juga bisa digunakan untuk menyamarkan ujaran kebencian. Penelitian ini menyelidiki peran emoji dalam klasifikasi ujaran kebencian dalam Bahasa Indonesia, dengan memanfaatkan model IndoBERT yang diperkaya melalui **emoji description**. 

Selain itu, model multibahasa **MBERT** juga digunakan untuk membandingkan efek translasi terhadap performa klasifikasi.

---

## 🎯 Tujuan Penelitian

✅ Menganalisis peran emoji dalam memperkuat atau mengubah konteks ujaran kebencian.  
✅ Mengevaluasi dampak penggunaan deskripsi emoji terhadap kinerja model IndoBERT.  
✅ Mengeksplorasi pengaruh hyperparameter (batch size, learning rate, epoch) terhadap performa model.  
✅ Membandingkan hasil klasifikasi dengan MBERT sebagai baseline terhadap data asli berbahasa Inggris.

---

## 🧾 Dataset

Data yang digunakan adalah tweet berbahasa Inggris yang diterjemahkan ke dalam Bahasa Indonesia. Dataset berasal dari:

- 🐍 **HateEmoji** – dari christophsonntag  
- 🕊 **TweetEval** – Hate & Offensive (oleh CardiffNLP)  
- 🧨 **Emoji-Based Hate Speech** – oleh Sushil Dalavi et al.

Tiap dataset diuji dalam dua kondisi:
- Teks asli dengan emoji
- Teks dengan emoji yang diubah menjadi deskripsi via `emoji.demojize()`

---

## ⚙️ Teknologi yang Digunakan

| Komponen            | Teknologi                          |
|---------------------|------------------------------------|
| Model NLP           | IndoBERT-liteBASE, MBERT           |
| Tokenisasi          | WordPiece (HuggingFace Transformers) |
| Emoji Handling      | `emoji.demojize()` (Python)        |
| Text Processing     | Sastrawi (Stemming, Stopwords)     |
| Resampling          | Undersampling, SMOTE               |
| Evaluasi            | Accuracy, Precision, Recall, F1-Score |

---

## 🔧 Hyperparameter Eksperimen

- **Batch size**: `16`, `32`
- **Learning rate**: `2e-5`, `3e-5`, `5e-5`
- **Epoch**: `2`, `3`, `4`

---

## 🧪 Desain Eksperimen

Eksperimen terbagi dalam 4 skenario utama:

1. 💬 IndoBERT + Emoji
2. ✂️ IndoBERT tanpa Emoji
3. 🌐 MBERT + Emoji
4. 🚫 MBERT tanpa Emoji

---

## 👩‍💻 Tim Peneliti

- **Walker Valentinus Simanjuntak** – 12S21012  
- **Ruth Marelisa Hutagalung** – 12S21046  
- **Lamria Magdalena Tampubolon** – 12S21055  

📍 Program Studi Sistem Informasi  
📚 Fakultas Informatika dan Teknik Elektro  
🏫 Institut Teknologi Del, 2025

---
