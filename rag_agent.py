"""RAG Support Agent — inti.

Menjawab pertanyaan pelanggan hanya dari dokumen yang diberikan, dengan sitasi.
Tanpa dependensi eksternal (Python stdlib saja).

Dua mode jawaban:
  - ekstraktif (default): mengembalikan potongan dokumen paling relevan + sitasi.
    Jalan tanpa API key apa pun.
  - LLM (opsional): bila env OPENAI_API_KEY diset, jawaban dirangkai model dari
    konteks yang ter-retrieve saja, dengan instruksi eksplisit untuk tidak
    mengarang dan mencantumkan sitasi. Gagal -> jatuh kembali ke ekstraktif.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# Stopword kecil Bahasa Indonesia + Inggris — cukup untuk retrieval demo.
STOPWORDS = set("""
yang dan di ke dari untuk dengan adalah ini itu pada sebagai juga atau jika
apakah bisa saya kami anda kamu kita mereka dia nya lah kah pun nya the a an
of to in on for with is are was were be by at as it this that from or and
""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Skor minimum cosine similarity agar jawaban dianggap ditemukan.
# Di bawah ini agent menjawab jujur "tidak ditemukan", bukan mengarang.
DEFAULT_THRESHOLD = 0.05


def _stem(token: str) -> str:
    """Stemmer ringan Bahasa Indonesia (tanpa kamus): cukup untuk mencocokkan
    varian kata seperti melacak/lacak dan pengiriman/dikirim. Diterapkan
    konsisten di sisi dokumen maupun pertanyaan, jadi pencocokan tetap adil."""
    t = token
    for suffix in ("kan", "an", "nya", "lah", "kah", "pun"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 3:
            t = t[: -len(suffix)]
            break
    for prefix in ("meng", "meny", "mem", "men", "me", "ber", "ter", "di", "ke", "se", "per", "pe"):
        if t.startswith(prefix) and len(t) - len(prefix) >= 3:
            t = t[len(prefix):]
            break
    if t.endswith("i") and len(t) > 4:
        t = t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return [_stem(t) for t in tokens if t not in STOPWORDS and len(t) >= 2]


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class Hit:
    doc: Document
    score: float


@dataclass
class Answer:
    question: str
    text: str
    citations: list[str]
    found: bool
    mode: str  # "extractive" atau "llm"
    hits: list[Hit] = field(default_factory=list)


class RagAgent:
    def __init__(self, docs: list[Document], threshold: float = DEFAULT_THRESHOLD):
        self.docs = docs
        self.threshold = threshold
        self._build_index()

    # ---------- index TF-IDF ----------

    def _build_index(self) -> None:
        self.doc_tokens: list[list[str]] = [tokenize(d.text) for d in self.docs]
        df: dict[str, int] = {}
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = max(len(self.docs), 1)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.df = df
        self.doc_vecs: list[dict[str, float]] = []
        for toks in self.doc_tokens:
            vec: dict[str, float] = {}
            for t in toks:
                vec[t] = vec.get(t, 0.0) + 1.0
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.doc_vecs.append(
                {t: (count / norm) * self.idf.get(t, 1.0) for t, count in vec.items()}
            )

    def _query_vec(self, question: str) -> dict[str, float]:
        vec: dict[str, float] = {}
        for t in tokenize(question):
            vec[t] = vec.get(t, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: (c / norm) * self.idf.get(t, 1.0) for t, c in vec.items()}

    def retrieve(self, question: str, top_k: int = 3) -> list[Hit]:
        q = self._query_vec(question)
        scored: list[Hit] = []
        for doc, vec in zip(self.docs, self.doc_vecs):
            score = sum(q.get(t, 0.0) * w for t, w in vec.items())
            scored.append(Hit(doc, score))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    # ---------- jawaban ----------

    def answer(self, question: str, top_k: int = 3) -> Answer:
        hits = self.retrieve(question, top_k=top_k)
        # Aturan kejujuran: skor saja tidak cukup pada korpus kecil — satu kata
        # langka yang cocok bisa menghasilkan cosine tinggi padahal pertanyaan
        # tidak berhubungan (mis. "Siapa presiden Indonesia?" cocok kata
        # "Indonesia" saja). Syarat ditemukan: skor melewati threshold, minimal
        # 2 kata kunci pertanyaan muncul di dokumen teratas, DAN minimal satu
        # kata kunci itu spesifik (muncul di <= 2 dokumen) — supaya pertanyaan
        # yang hanya berbagi kata generik ("Toko Maju", "hari") tetap ditolak.
        matched_terms: set[str] = set()
        if hits and hits[0].score > 0:
            matched_terms = set(tokenize(question)) & set(tokenize(hits[0].doc.text))
        specific = any(self.df.get(t, 99) <= 2 for t in matched_terms)
        if (not hits or hits[0].score < self.threshold
                or len(matched_terms) < 2 or not specific):
            return Answer(
                question=question,
                text="Maaf, jawaban untuk pertanyaan itu tidak ditemukan di dokumen "
                     "perusahaan. Silakan hubungi tim support manusia untuk bantuan "
                     "lebih lanjut.",
                citations=[],
                found=False,
                mode="extractive",
                hits=hits,
            )
        if os.environ.get("OPENAI_API_KEY"):
            llm_text = self._llm_answer(question, hits)
            if llm_text:
                return Answer(
                    question=question,
                    text=llm_text,
                    citations=[h.doc.doc_id for h in hits if h.score > 0],
                    found=True,
                    mode="llm",
                    hits=hits,
                )
        # Mode ekstraktif: potongan dokumen teratas + sitasi eksplisit.
        parts = []
        for h in hits:
            if h.score <= 0:
                continue
            snippet = " ".join(h.doc.text.split())[:400]
            parts.append(f"[{h.doc.doc_id}] {h.doc.title}: {snippet}")
        return Answer(
            question=question,
            text="\n\n".join(parts),
            citations=[h.doc.doc_id for h in hits if h.score > 0],
            found=True,
            mode="extractive",
            hits=hits,
        )

    def _llm_answer(self, question: str, hits: list[Hit]) -> str | None:
        context = "\n\n".join(
            f"[{h.doc.doc_id}] {h.doc.title}:\n{h.doc.text}" for h in hits
        )
        payload = {
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Kamu agen customer support. Jawab HANYA berdasarkan "
                        "dokumen di bawah. Cantumkan sitasi [doc_id] untuk setiap "
                        "fakta. Jika jawaban tidak ada di dokumen, katakan "
                        "terus terang bahwa informasinya tidak ditemukan. "
                        "Jangan mengarang."
                    ),
                },
                {
                    "role": "user",
                    "content": f"DOKUMEN:\n{context}\n\nPERTANYAAN: {question}",
                },
            ],
        }
        try:
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None  # panggil gagal -> pemanggil jatuh ke mode ekstraktif


def load_docs(docs_dir: str | Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        title = lines[0].lstrip("# ").strip() if lines else path.stem
        docs.append(Document(doc_id=path.stem, title=title, text=raw))
    return docs
