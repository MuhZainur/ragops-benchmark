# ⚙️ HYPERPARAMETERS & TUNING GUIDE

Dokumen ini adalah **panduan tuning parameter tingkat lanjut** untuk arsitektur RAG pada domain dokumen finansial enterprise.

---

## 1. Chunking Hyperparameters

### A. `chunk_size` (Tokens)
* **Under-chunking (< 200 tokens):**
  * *Bahaya:* Kalimat penjelasan angka keuangan terpotong. Header tabel akuntansi hilang dari baris angka.
  * *Kelebihan:* Kemiripan vektor semantik sangat tajam, retrieval precision tinggi.
* **Over-chunking (> 1500 tokens):**
  * *Bahaya:* *Lost-in-the-middle* meningkat. Dilusi semantik (embedding vektor menjadi rata-rata dari banyak topik sekaligus, sehingga gagal menangkap query spesifik).
  * *Kelebihan:* Integritas konteks tabel dan narasi terjaga.
* **Rekomendasi RAG-Ops:**
  * **Naive Baseline:** 512 tokens.
  * **Hierarchical (Small-to-Big):** Child Index = **128 tokens**, Parent Context = **1024 tokens**.

### B. `chunk_overlap` (Tokens)
* **Rekomendasi:** **10% – 15%** dari `chunk_size` (sekitar 50 – 75 tokens).
* **Fungsi:** Mencegah klaim penting terbelah di batas potongan chunk. Misalnya, frasa *"Laba bersih naik 20% namun dipotong oleh pajak tangguhan"* tidak boleh terputus sebelum kata *"namun"*.

---

## 2. Sparse Search (BM25) Tuning

Formula BM25 memiliki dua parameter utama:

| Parameter | Default | Rekomendasi Finansial | Penjelasan Teknis |
| :--- | :--- | :--- | :--- |
| **$k_1$ (Term Saturation)** | 1.2 | **1.5 – 1.8** | Menentukan seberapa cepat bobot kata jenuh ketika kata tersebut diulang berkali-kali dalam satu bab 10-K. Di laporan keuangan, istilah seperti *"Risk Factors"* sering berulang, sehingga $k_1 = 1.6$ ideal untuk mencegah saturasi instan. |
| **$b$ (Document Length Normalization)** | 0.75 | **0.75** | Menghukum dokumen/chunk yang panjang agar tidak selalu menang melawan chunk pendek yang padat fakta. Nilai $0.75$ adalah standar emas industri. |

---

## 3. Fusion & Re-Ranking Parameters

### A. RRF Smoothing Constant ($k$)
* **Formula:** $\text{RRF}(d) = \frac{1}{k + \text{rank}(d)}$
* **Pilihan Nilai:**
  * $k = 10$: Memberikan bobot sangat agresif pada dokumen peringkat #1. Jika Dense atau BM25 salah di peringkat pertama, hasilnya langsung bias.
  * **$k = 60$ (Rekomendasi Standar Cormack et al.):** Memberikan gradien bobot yang stabil antara peringkat 1 sampai 10. Dokumen yang muncul di peringkat 3 di Dense dan peringkat 4 di BM25 akan mendapatkan skor kombinasi yang adil dan kuat.

### B. Retrieval Funnel: `top_k_retrieval` vs `rerank_top_n`
* **Prinsip Funnel Dua Tahap:**
  1. **Tahap 1 (Broad Recall):** Ambil **20 chunk** dari gabungan Hybrid RRF (BM25 + Vector).
  2. **Tahap 2 (Precision Filter):** Kirim 20 chunk tersebut ke Cross-Encoder FlashRank. Ambil hanya **3 hingga 5 chunk teratas**.
* **`score_threshold` (Cutoff):**
  * Tetapkan threshold **$0.35$**. Jika dari 20 chunk hanya ada 2 chunk yang skor Cross-Encoder-nya $\ge 0.35$, maka sistem hanya mengirim 2 chunk ke LLM. Ini menghemat token dan memangkas risiko halusinasi akibat context sampah.

---

## 4. Generation Hyperparameters

| Parameter | Nilai Finansial | Alasan Pemilihan |
| :--- | :--- | :--- |
| `temperature` | **0.0** | Nol kreativitas. Model hanya diizinkan mereproduksi fakta yang terverifikasi di context. |
| `top_p` | **0.9** | Nucleus sampling ketat untuk memilih token dengan probabilitas tertinggi. |
| `max_tokens` | **1024** | Cukup untuk sintesis analisis keuangan 3 paragraf lengkap dengan sitasi pasal/tabel. |
