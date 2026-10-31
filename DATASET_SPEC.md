# 📁 DATASET SPECIFICATION: SEC 10-K & GOLDEN BENCHMARK

Dokumen ini mendefinisikan korpus data, pipeline pemrosesan dokumen keuangan, dan struktur dataset pengujian.

---

## 1. Korpus Dokumen Target (SEC Form 10-K)

Proyek ini menggunakan dokumen laporan keuangan tahunan resmi (Form 10-K) periode **FY2023 - FY2024** dari 3 emiten utama:

1. **NVIDIA Corporation (NVDA) - FY2024:**
   * *Item 1 & 1A:* Deskripsi arsitektur GPU Hopper/Blackwell, rantai pasok TSMC, regulasi lisensi ekspor ke Tiongkok.
   * *Item 7 (MD&A):* Analisis pendapatan segmen Data Center, Gaming, Automotive, dan Gross Margin expansion.
   * *Item 8:* Laporan Neraca, Arus Kas, dan Catatan Kaki Liabilitas Komitmen.
2. **Apple Inc. (AAPL) - FY2024:**
   * *Item 1 & 1A:* Pertumbuhan ekosistem Services, gugatan antimonopoli Departemen Kehakiman AS (DoJ), ketergantungan manufaktur di Asia.
   * *Item 7 (MD&A):* Pertumbuhan segmen iPhone, Mac, Wearables, dan margin kotor hardware vs services.
3. **Microsoft Corporation (MSFT) - FY2024:**
   * *Item 1 & 1A:* Integrasi Azure AI, kemitraan strategis OpenAI, belanja modal pusat data.
   * *Item 7 (MD&A):* Pertumbuhan Intelligent Cloud, Productivity & Business Processes.

---

## 2. Ingestion & Preprocessing Pipeline (`src/ingestion/parser.py`)

Laporan SEC 10-K mentah sering kali berupa file HTML raksasa (5MB - 20MB) dengan tag tabel yang berlapis-lapis.

### Langkah Normalisasi:
1. **Pembersihan Boilerplate:** Menghilangkan tag HTML non-semantik (`<font>`, `<span>`, style CSS berulang).
2. **Preservasi Tabel ke Markdown:** Mengonversi tabel HTML menjadi sintaks Markdown Pipe Table standar (`| Header 1 | Header 2 |`) agar LLM dapat memahami relasi baris dan kolom.
3. **Section Boundary Detection:** Mengidentifikasi secara otomatis batas-batas bagian resmi SEC:
   * `ITEM 1. Business`
   * `ITEM 1A. Risk Factors`
   * `ITEM 7. Management's Discussion and Analysis (MD&A)`
   * `ITEM 8. Financial Statements and Supplementary Data`
4. **Metadata Enrichment:** Setiap chunk yang dihasilkan otomatis diberi metadata:
   ```json
   {
     "ticker": "NVDA",
     "fiscal_year": 2024,
     "item_section": "ITEM_7_MDA",
     "chunk_type": "table_with_narration",
     "parent_id": "NVDA_2024_MDA_PARENT_014"
   }
   ```

---

## 3. Komposisi Golden Benchmark Dataset (50 Pertanyaan)

Agar pengujian objektif dan valid, dataset uji dibagi menjadi 3 kategori:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   50 GOLDEN BENCHMARK QUESTIONS                        │
├───────────────────┬───────────────────┬────────────────────────────────┤
│ 1. SINGLE-HOP     │ 2. MULTI-HOP      │ 3. ADVERSARIAL                 │
│    NUMERICAL (20) │    COMPARATIVE(15)│    UNANSWERABLE (15)           │
├───────────────────┼───────────────────┼────────────────────────────────┤
│ Pertanyaan fakta  │ Membandingkan dua │ Pertanyaan yang informasinya   │
│ angka langsung.   │ emiten sekaligus  │ TIDAK ADA di 10-K. Menguji     │
│ Contoh: CapEx     │ (Apple vs Nvidia) │ apakah RAG berani menolak      │
│ Nvidia FY24.      │ atau dua tahun    │ menjawab atau halusinasi.      │
│                   │ berbeda.          │                                │
└───────────────────┴───────────────────┴────────────────────────────────┘
```
