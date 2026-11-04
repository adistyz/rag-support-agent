"""CLI demo: python demo.py "pertanyaan"  atau  python demo.py (interaktif)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_agent import RagAgent, load_docs

HERE = Path(__file__).parent


def main() -> None:
    agent = RagAgent(load_docs(HERE / "docs"))
    questions = sys.argv[1:]
    if questions:
        ask(agent, " ".join(questions))
        return
    print("RAG Support Agent — Toko Maju (ketik 'keluar' untuk berhenti)")
    while True:
        try:
            q = input("\nPelanggan: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"keluar", "exit", "quit"}:
            break
        if q:
            ask(agent, q)


def ask(agent: RagAgent, question: str) -> None:
    ans = agent.answer(question)
    print(f"\nAgent [{ans.mode}]: {ans.text}")
    if ans.citations:
        print("Sumber:", ", ".join(ans.citations))
    else:
        print("Sumber: tidak ada (jawaban tidak ditemukan di dokumen)")


if __name__ == "__main__":
    main()
