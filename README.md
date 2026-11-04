# RAG Support Agent — Agen Customer Support berbasis Dokumen

Agen support yang menjawab pertanyaan pelanggan **hanya dari dokumen perusahaan sendiri**,
dengan sitasi sumber. Kalau jawabannya tidak ada di dokumen, agent bilang terus terang
tidak tahu — bukan mengarang.

## Masalah yang diselesaikan

UMKM dan bisnis online kebanjiran pertanyaan yang jawabannya sebenarnya sudah ada di
FAQ, kebijakan pengiriman, retur, dan garansi. Agent ini menjawab otomatis, konsisten,
dan bisa ditelusur sumbernya — sehingga pemilik usaha tidak perlu menjawab hal yang
sama berulang kali, dan pelanggan tidak menunggu jam operasional.

## Cara kerja

```
Pertanyaan pelanggan
      │
      ▼
  Tokenisasi + TF-IDF ──► Retrieval (cosine similarity, top-3)
      │
      ▼
  Skor teratas < threshold? ──► "Maaf, tidak ditemukan di dokumen" (jujur, tanpa karangan)
      │ tidak
      ▼
  Jawaban + sitasi [doc_id]
      │
      ├── Mode default: ekstraktif (tanpa API key, jalan 100% lokal, stdlib saja)
      └── Mode LLM (opsional): OPENAI_API_KEY diset → jawaban dirangkai model dari
          konteks yang ter-retrieve saja, tetap dengan sitasi
```

## Struktur

```
rag-support-agent/
├── rag_agent.py     # inti: chunk, TF-IDF, retrieval, jawaban, mode LLM opsional
├── docs/            # dokumen demo (toko online fiktif "Toko Maju", Bahasa Indonesia)
├── demo.py          # CLI: python demo.py "pertanyaan pelanggan"
├── eval.py          # evaluasi: hit-rate retrieval, latensi, kecocokan fakta
└── requirements.txt # tanpa dependensi (Python stdlib saja)
```

## Menjalankan

```bash
# Tanpa API key apa pun — mode ekstraktif langsung jalan:
python demo.py "Berapa lama pengiriman ke Jakarta?"

# Mode interaktif:
python demo.py

# Evaluasi (10 pertanyaan uji, metrik riil):
python eval.py
```

## Mode LLM (opsional)

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"   # opsional, default gpt-4o-mini
python demo.py "Apakah bisa retur barang?"
```

Prompt sistem mode LLM memerintahkan model menjawab **hanya dari konteks yang
diberikan** dan mencantumkan sitasi. Kalau konteks tidak memuat jawaban, model
diperintah menjawab tidak tahu. Bila panggilan LLM gagal, agent otomatis jatuh
kembali ke mode ekstraktif.

## Hasil evaluasi (diukur nyata, bukan klaim)

Perintah: `python eval.py` — 10 pertanyaan uji atas 7 dokumen demo, mode
ekstraktif (tanpa API key), di mesin ini:

- **Top-1 hit rate: 100% (10/10)** — dokumen yang benar ada di peringkat 1
- **Top-3 hit rate: 100% (10/10)**
- **Fakta terjawab: 100% (10/10)** — jawaban mengandung fakta kunci yang diharapkan
- **Latensi: mean 0,2 ms, p50 0,2 ms, p95 0,2 ms** per pertanyaan
- **Uji kejujuran: 3/3 pertanyaan di luar dokumen ditolak dengan benar**
  ("Siapa presiden Indonesia?", "Apakah ada diskon untuk pembelian 100 unit?",
  "Berapa harga saham Toko Maju?") — agent menjawab tidak ditemukan, bukan mengarang.

Kejujuran soal angka ini: korpus demo kecil (7 dokumen, 10 pertanyaan yang
dibuat dari dokumen itu sendiri), jadi 100% di sini adalah sinyal demo, bukan
jaminan akurasi produksi. Yang lebih berarti untuk klien adalah perilaku
fallback-nya: agent menolak menjawab di luar dokumen alih-alih mengarang.

## Keterbatasan yang diakui (dan jalur upgrade-nya)

Retrieval saat ini berbasis kata kunci (TF-IDF + stemmer ringan Bahasa
Indonesia). Ia tidak memahami makna seperti embedding — sinonim yang tidak
berbagi kata (mis. "ongkir" vs "biaya kirim") bisa lolos. Itu sebabnya
aturan kejujuran di atas (threshold + minimal 2 kata cocok + minimal 1 kata
spesifik) ada. Jalur upgrade yang bisa ditagih ke klien: model embedding,
re-ranking, riwayat percakapan, dan channel WhatsApp/Telegram.

## Kenapa ini portofolio yang bisa dijual

- **Pembeli jelas:** setiap bisnis dengan FAQ/dokumen adalah calon klien.
- **Demo 5 menit:** ganti folder `docs/` dengan dokumen klien, agent langsung bisa
  ditanya — tanpa training ulang.
- **Risiko rendah:** tidak ada eksekusi transaksi, tidak ada akses wallet, tidak
  ada aksi yang tidak bisa dibatalkan. Satu-satunya output adalah teks + sitasi.
- **Jalur upgrade nyata:** tambah embedding model, re-ranking, riwayat chat,
  dan channel WhatsApp/Telegram — semuanya bisa ditagih sebagai fase berikutnya.
