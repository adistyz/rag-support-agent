"""Evaluasi RAG Support Agent: 10 pertanyaan uji dengan metrik riil.

Metrik:
  - top1_hit: dokumen yang diharapkan ada di peringkat 1
  - top3_hit: dokumen yang diharapkan ada di 3 teratas
  - fact_hit: jawaban mengandung fakta kunci yang diharapkan
  - latensi: mean / p50 / p95 per pertanyaan (milidetik)
Hasil ditulis ke results.json dan dicetak sebagai tabel.
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_agent import RagAgent, load_docs

HERE = Path(__file__).parent

# (pertanyaan, doc_id yang diharapkan, fakta kunci yang harus ada di jawaban)
CASES = [
    ("Berapa lama pengiriman ke Jakarta?", "pengiriman", "1-2 hari"),
    ("Apakah bisa retur barang?", "retur", "7 hari"),
    ("Garansi produk elektronik berapa lama?", "garansi", "12 bulan"),
    ("Metode pembayaran apa saja yang diterima?", "pembayaran", "transfer bank"),
    ("Jam berapa customer service buka?", "operasional", "08:00"),
    ("Kapan ongkir gratis berlaku?", "pengiriman", "500"),
    ("Bagaimana cara melacak pesanan saya?", "pelacakan", "resi"),
    ("Apakah bisa bayar di tempat (COD)?", "pembayaran", "COD"),
    ("Barang sampai dalam keadaan rusak, apa yang harus saya lakukan?", "retur", "rusak"),
    ("Apakah Toko Maju mengirim ke luar pulau?", "pengiriman", "seluruh Indonesia"),
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return s[idx]


def main() -> None:
    agent = RagAgent(load_docs(HERE / "docs"))
    rows = []
    latencies = []
    for question, expected_doc, expected_fact in CASES:
        t0 = time.perf_counter()
        ans = agent.answer(question)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        top_docs = [h.doc.doc_id for h in ans.hits]
        rows.append({
            "question": question,
            "expected_doc": expected_doc,
            "top1": top_docs[0] if top_docs else None,
            "top3": top_docs,
            "top1_hit": bool(top_docs) and top_docs[0] == expected_doc,
            "top3_hit": expected_doc in top_docs,
            "fact_hit": expected_fact.lower() in ans.text.lower(),
            "found": ans.found,
            "mode": ans.mode,
            "latency_ms": round(ms, 1),
        })

    n = len(rows)
    summary = {
        "n_questions": n,
        "mode": rows[0]["mode"] if rows else None,
        "top1_hit_rate": round(sum(r["top1_hit"] for r in rows) / n, 3),
        "top3_hit_rate": round(sum(r["top3_hit"] for r in rows) / n, 3),
        "fact_hit_rate": round(sum(r["fact_hit"] for r in rows) / n, 3),
        "latency_mean_ms": round(statistics.mean(latencies), 1),
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
        "rows": rows,
    }
    (HERE / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"{'Pertanyaan':<58} {'top1':<12} {'top3':<5} {'fakta':<6} {'ms':>6}")
    for r in rows:
        q = r["question"][:56]
        print(f"{q:<58} {str(r['top1']):<12} "
              f"{'ya' if r['top3_hit'] else 'TIDAK':<5} "
              f"{'ya' if r['fact_hit'] else 'TIDAK':<6} {r['latency_ms']:>6}")
    print()
    print(f"Top-1 hit rate : {summary['top1_hit_rate']:.0%} ({sum(r['top1_hit'] for r in rows)}/{n})")
    print(f"Top-3 hit rate : {summary['top3_hit_rate']:.0%} ({sum(r['top3_hit'] for r in rows)}/{n})")
    print(f"Fakta terjawab : {summary['fact_hit_rate']:.0%} ({sum(r['fact_hit'] for r in rows)}/{n})")
    print(f"Latensi        : mean {summary['latency_mean_ms']} ms, "
          f"p50 {summary['latency_p50_ms']} ms, p95 {summary['latency_p95_ms']} ms")
    print("Mode           :", summary["mode"], "(tanpa API key)")


if __name__ == "__main__":
    main()
